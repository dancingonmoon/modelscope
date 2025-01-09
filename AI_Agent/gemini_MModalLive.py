import asyncio
from google import genai
import logging
import traceback
import pyaudio
import mss
from mss import tools
from PIL import Image
import io
import base64
import webrtcvad
import numpy as np


CHANNELS = 1
RECEIVE_SAMPLE_RATE = 16000  # 输入音频格式：16kHz 小端字节序的原始 16 位 PCM 音频
SEND_SAMPLE_RATE = 24000  # 输出音频格式：24kHz 小端字节序的原始 24 位 PCM 音频
CHUNK_SIZE = 1024
FORMAT = 2  # paInt16, 16bit

MODEL = "models/gemini-2.0-flash-exp"
# Multimodal Live API 支持以下语音：Aoede、Charon、Fenrir、Kore 和 Puck;
VOICE_NAME = ["Aoede", "Charon", "Fenrir", "Kore", "Puck"][3]
SPEECH_CONFIG = {
    "voice_config": {"prebuilt_voice_config": {"voice_name": f"{VOICE_NAME}"}}
}
CONFIG = {
    "generation_config": {
        "response_modalities": ["AUDIO"],
        "speech_config": SPEECH_CONFIG,
    }
}
# CONFIG = {"generation_config": {"response_modalities": ["AUDIO"]}}
client = genai.Client(
    http_options={"api_version": "v1alpha"}
)  # api_key直接从环境变量中名称为GOOGLE_API_KEY获取

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        # logging.FileHandler("app.log")  # 输出到文件
    ],
)


async def txt2txt():
    """
    gemini 2.0 multi-modal live api simple text to text with google_search
    """
    search_tool = {"google_search": {}}
    config = {"response_modalities": ["TEXT"], "tools": [search_tool]}
    model_id = "gemini-2.0-flash-exp"
    async with client.aio.live.connect(model=model_id, config=config) as session:
        while True:
            message = input("User> ")
            if message.lower() == "exit":
                break
            await session.send(message, end_of_turn=True)

            async for response in session.receive():
                if response.text is None:
                    continue
                print(response.text, end="")


class GeminiLiveStream:
    """
    implements the interaction with the Live API：
    run - The main loop
    This method:
    Opens a websocket connecting to the Live API.Calls the initial setup method.
    Then enters the main loop where it alternates between send and recv until send returns False.
    send - Sends input text to the api, as screen snap video as well
    The send method collects input text from the user, wraps it in a client_content message, and sends it to the model.
    If the user sends a q this method returns False to signal that it's time to quit.
    recv - Collects audio from the API and plays it
    The recv method collects audio chunks in a loop. It breaks out of the loop once the model sends a turn_complete method, and then plays the audio.
    """

    def __init__(
        self,
        config=None,
        pyaudio_instance: pyaudio.PyAudio = None,
        logger: logging.Logger = None,
        model: str = "models/gemini-2.0-flash-exp",
    ):
        self.model = model
        if config is None:
            config = {"generation_config": {"response_modalities": ["AUDIO"]}}
        self.config = config

        if pyaudio_instance is None:
            pyaudio_instance = pyaudio.PyAudio()
        self.pyaudio_instance = pyaudio_instance

        self.session = None
        self.audio_out_queue = None  # 缓存,模型audio 输出;
        self.in_queue = None  # 缓存,模型输入: microphone audio, screen video, camera video,
        self.pause_stream = False
        # self.stop_stream = False
        self.stop_stream = asyncio.Event()  # 创建等待事件对象

        if logger is None:
            logger = logging.getLogger("Pyaudio_Record_Player")
            logger.setLevel("INFO")
        self.logger = logger
        self.logger.info(f"stop_stream:{self.stop_stream.is_set()}")

    async def run(
        self,
        sample_width: int = 2,
        channels: int = 1,
        in_rate: int = 16000,
        out_rate: int = 24000,
        in_chunk_size: int = 480,
        vad_mode: int = 2,
    ):
        self.logger.debug("connect")
        try:
            async with (
                client.aio.live.connect(
                    model=self.model, config=self.config
                ) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_out_queue = asyncio.Queue()  # 缓存,模型audio 输出;
                self.in_queue = asyncio.Queue(
                    maxsize=5
                )  # 缓存,模型输入: microphone audio, screen video, camera video,
                send_txt_task = tg.create_task(self.send())
                tg.create_task(self.send_realtime())
                tg.create_task(
                    self.microphone_audio(
                        sample_width, channels, in_rate, in_chunk_size, vad_mode
                    )
                )
                tg.create_task(self.recv())
                tg.create_task(self.play_audio(sample_width, channels, out_rate))

                # 此处等待send_txt_task结束，而实际上self.send()函数是无限循环的input,除非user输入了exit，主动退出；
                # self.send()函数中的break，会退出self.send()函数，而不会退出async with 语句块;
                # 所以，当await send_txt_task 等待完成，即用户请求退出，需要将这个异步进程，以及后面的task都要cancel，
                # 所以，手动raise asyncio.CancelledError("User requested exit")
                await send_txt_task
                raise asyncio.CancelledError("User requested exit")

                # 当exit_event被设置时，TaskGroup会等待所有任务完成，并自动取消所有任务;当有
                # exit_request等待事件处于等待（block），当set时，该等待进程被唤醒，block解除
                # TaskGroup的目的是管理一组相互依赖的任务，这些任务应该一起启动和结束。
                # 当一个任务完成时，TaskGroup认为整个任务组的工作已经完成，因此会尝试取消其他所有任务
                # await self.stop_stream.wait()

        except asyncio.CancelledError:
            logger.info("asyncio.CancelledError")
        except ExceptionGroup as EG:
            traceback.print_exception(EG)

    async def send(self):
        """
        持续不断的user input提示,直到q/quit/exit退出
        """
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text in ["pause", "p"]:
                self.pause_stream = True
                self.stop_stream.clear()
                self.logger.info(f"User input: {text}")
            elif text in ["c", "continue"]:
                self.pause_stream = False
                self.stop_stream.clear()
                self.logger.info(f"User input: {text}")
            elif text in ["q", "quit"]:
                self.pause_stream = False
                self.stop_stream.set()
                self.logger.info(f"User input: {text}")
                break
            else:
                await self.session.send(text or ".", end_of_turn=True)
            # await asyncio.sleep(0.1)

    async def recv(self):
        """
        Background task to reads from the websocket and write pcm chunks to the output queue
        """
        while not self.stop_stream.is_set():
            self.logger.debug("receive")
            # read chunks from the socket
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_out_queue.put_nowait(data)
                    # await self.audio_out_queue.put(data) # 使用await，可以保证，put操作是同步的，不会出现阻塞;
                else:
                    self.logger.debug(f"Unhandled server message! - {response}")
                    if text := response.text:
                        print(text, end="")

            # If you interrupt the model, it sends a turn_complete.For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded much more audio than has played yet.
            # 模型本身是支持被打断，即上一个回答还在持续输出的时候，提出新问题，模型会中止上个提问的输出，开始新问题的输出；
            # 所以，self.audio_queue队列中缓存的之前的回答需要清除，否则，就会将缓存中所有内容都播放
            # 因为是模型被打断的时候，会是一个新的输出，所以，在async for循环外面,不断重复的取出(移出)最后一个数据,直到队列为空
            # 潜在的问题也是:当for response in turn循环完毕,播放没有完成的话,后面的数据会被自动empty掉
            while not self.audio_out_queue.empty():  # 当加上这句块，有掉字的情况
                self.audio_out_queue.get_nowait()
                # await self.audio_out_queue.get()

    async def play_audio(
        self,
        sample_width: int = 2,
        channels: int = 1,
        rate: int = 24000,
    ):
        stream = await asyncio.to_thread(
            self.pyaudio_instance.open,
            format=self.pyaudio_instance.get_format_from_width(sample_width),
            channels=channels,
            rate=rate,
            output=True,
        )

        while not self.stop_stream.is_set():
            if self.pause_stream:
                await asyncio.sleep(0.1)  # 避免运行因长循环，滞留在此处，导致user_command阻塞
                continue
            audio_data = await self.audio_out_queue.get()
            # if audio_data is None:
            #     self.stop_stream = True
            #     stream.stop_stream()
            #     stream.close()
            #     self.logger.info("音频播放结束")
            #     break
            await asyncio.to_thread(stream.write, audio_data)
        if self.stop_stream:
            stream.stop_stream()
            stream.close()
            self.logger.info("user_command终止")

    def _get_screen(self):
        with mss.mss() as sct:
            # monitor = sct.monitors[0]
            # i = sct.grab(monitor)  # 捕获指定显示器的截图
            screen_shot = sct.shot()  # 捕获活动屏幕的截图

        image_bytes = mss.tools.to_png(screen_shot.rgb, screen_shot.size)  # raw data
        img = Image.open(io.BytesIO(image_bytes))  # image data bytes
        # 写入 文件格式的内存对象
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        # 从内存对象中读出，
        image_bytes = image_io.read()
        mime_type = "image/jpeg"
        return {
            "mime_type": mime_type,
            "data": base64.b64encode(image_bytes).decode(),
        }

    async def get_screen(self):
        self.logger.info(f"stop_stream:{self.stop_stream.is_set()}")
        while not self.stop_stream.is_set():
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                self.logger.info("failed to get screen")
                break
            await asyncio.sleep(1.0)
            await self.in_queue.put(frame)

    async def send_realtime(self):
        while self.stop_stream.is_set():
            msg = await self.in_queue.get()
            await self.session.send(msg)

    async def microphone_audio(
        self,
        sample_width: int = 2,
        channels: int = 1,
        rate: int = 16000,
        chunk_size: int = 480,  # 16Khz, 30ms长度,对应的帧长度是480
        vad_mode: int = 2,  # vad 模式，0-3，3最敏感
    ):
        """
        持续不断的从麦克风读取音频数据;使用asyncio.Queue来缓存队列,传递异步进程的音频数据,音频输入输出更加光滑;
        实现了回声抑制(经过vad判断is_speech,静音填充) [注: 可能有vad算法的原因,回声抑制效果不是最佳,可以考虑更换回声抑制算法库]
        """
        vad = webrtcvad.Vad(vad_mode)
        mic_info = self.pyaudio_instance.get_default_input_device_info()
        audio_stream = await asyncio.to_thread(
            self.pyaudio_instance.open,
            format=self.pyaudio_instance.get_format_from_width(sample_width),
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=chunk_size,
        )
        if __debug__:  # 这是一个条件语句，用于检查当前是否处于调试模式。
            kwargs = {
                "exception_on_overflow": False
            }  # 如果处于调试模式，kwargs 被设置为一个字典，这意味着在调试模式下，当音频流溢出（即缓冲区满，无法再接收更多数据）时，不会抛出异常。
        else:
            kwargs = {}  # 如果当前不是调试模式，这意味着在生产模式下，当音频流溢出时，可能会抛出异常，这通常是为了确保程序能够及时处理错误情况

        try:
            while not self.stop_stream.is_set():
                if self.pause_stream:
                    await asyncio.sleep(0.1)
                    continue
                try:
                    data = await asyncio.to_thread(
                        audio_stream.read, chunk_size, **kwargs
                    )
                    data_np = np.frombuffer(data, dtype=np.int16)
                    # 使用VAD检测是否是语音
                    is_speech = vad.is_speech(data_np.tobytes(), rate)
                    if is_speech:
                        audio_vad = data
                    else:
                        # 填补静音数据
                        silent_frame = np.zeros(chunk_size, dtype=np.int16)
                        audio_vad = silent_frame.tobytes()
                    await self.in_queue.put(
                        {"data": audio_vad, "mime_type": "audio/pcm"}
                    )
                except OSError as e:
                    self.logger.error(f"麦克风读取发生操作系统错误: {e}")
                    break  # 发生错误时退出循环
            if self.stop_stream.is_set():
                audio_stream.stop_stream()
                audio_stream.close()
                self.logger.info("麦克风停止录音")
        except ExceptionGroup as EG:
            traceback.print_exception(EG)
            self.logger.info(f"麦克风初始化或读取发生错误: {EG}")


if __name__ == "__main__":
    logger = logging.getLogger("gemini_MultiModal_Live")
    logger.setLevel("INFO")
    pya = pyaudio.PyAudio()
    # asyncio.run(txt2txt())
    geminiLive_instance = GeminiLiveStream(
        config=CONFIG, pyaudio_instance=pya, logger=logger
    )
    asyncio.run(
        geminiLive_instance.run(
            sample_width=2,
            channels=1,
            in_rate=16000,
            out_rate=24000,
            in_chunk_size=480,
            vad_mode=2,
        )
    )
