import asyncio
from google import genai
import wave
import logging
import traceback
import pyaudio

CHANNELS = 1
SEND_SAMPLE_RATE = 16000  # 输入音频格式：16kHz 小端字节序的原始 16 位 PCM 音频
RECEIVE_SAMPLE_RATE = 24000  # 输出音频格式：24kHz 小端字节序的原始 24 位 PCM 音频
CHUNK_SIZE = 1024

pya = pyaudio.PyAudio()
FORMAT = pya.get_format_from_width(2)  # paInt16, 16bit

MODEL = "models/gemini-2.0-flash-exp"
CONFIG = {"generation_config": {"response_modalities": ["AUDIO"]}}
client = genai.Client(
    http_options={"api_version": "v1alpha"}
)  # api_key直接从环境变量中名称为GOOGLE_API_KEY获取

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # 输出到控制台
                        # logging.FileHandler("app.log")  # 输出到文件
                    ])
logger = logging.getLogger("Gemini Live Stream")


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


async def async_enumerate(it):
    """
    对生成器迭代内容,进行异步的enumerate,即输出迭代内容的序列号,以及迭代内容本身
    :param it: 一个迭代器
    :return: 返回迭代的序列号,以及单次迭代内容
    """
    n = 0
    async for item in it:
        yield n, item
        n += 1


class GeminiLiveStream:
    """
    implements the interaction with the Live API：
    run - The main loop
    This method:
    Opens a websocket connecting to the Live API.Calls the initial setup method.
    Then enters the main loop where it alternates between send and recv until send returns False.
    send - Sends input text to the api
    The send method collects input text from the user, wraps it in a client_content message, and sends it to the model.
    If the user sends a q this method returns False to signal that it's time to quit.
    recv - Collects audio from the API and plays it
    The recv method collects audio chunks in a loop. It breaks out of the loop once the model sends a turn_complete method, and then plays the audio.
    """

    def __init__(self, config=None, pyaudio_instance: pyaudio.PyAudio = None, logger: logging.Logger = None):
        """ """
        if config is None:
            config = {"generation_config": {"response_modalities": ["AUDIO"]}}
        self.config = config

        if pyaudio_instance is None:
            pyaudio_instance = pyaudio.PyAudio()
        self.pyaudio_instance = pyaudio_instance

        self.session = None
        self.audio_out_queue = asyncio.Queue()  # 缓存,模型audio 输出;
        self.in_queue = asyncio.Queue()  # 缓存,模型输入: microphone audio, screen video, camera video,
        self.pause_stream = False
        self.stop_stream = False

        if logger is None:
            logger = logging.getLogger("Pyaudio_Record_Player")
            logger.setLevel("INFO")
        self.logger = logger


    async def run(self):
        logger.debug("connect")
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=self.config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_out_queue = asyncio.Queue()

                send_txt_task = tg.create_task(self.send())
                tg.create_task(self.recv())
                tg.create_task(self.play_audio())

                # 此处等待send_txt_task结束，而实际上self.send()函数是无限循环的input,除非user输入了exit，主动退出；
                # self.send()函数中的break，会退出self.send()函数，而不会退出async with 语句块;
                # 所以，当await send_txt_task 等待完成，即用户请求退出，需要将这个异步进程，以及后面的task都要cancell，
                # 所以，手动raise asyncio.CancelledError("User requested exit")
                await send_txt_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            logger.info("user requested exit")
        except ExceptionGroup as EG:
            traceback.print_exception(EG)

    async def send(self):
        """
        持续不断的user input提示,直到exit退出
        """
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() in ['q', 'quit', 'exit']:
                break
            await self.session.send(text or ".", end_of_turn=True)

    async def recv(self):
        """
        Background task to reads from the websocket and write pcm chunks to the output queue
        """
        while True:
            logger.debug("receive")
            # read chunks from the socket
            turn = self.session.receive()
            # async for n, response in async_enumerate(turn):
            #     logger.debug(f"got chunk: {str(response)}")
            async for response in turn:
                if data := response.data:
                    self.audio_out_queue.put_nowait(data)
                    # await self.audio_queue.put(data) # 使用await，可以保证，put操作是同步的，不会出现阻塞;
                else:
                    logger.debug(f"Unhandled server message! - {response}")
                    if text := response.text:
                        print(text, end="")

            #     if n == 0:
            #         print(
            #             response.server_content.model_turn.parts[
            #                 0
            #             ].inline_data.mime_type
            #         )
            #     print(".", end="")
            #
            # print("\n")

            # If you interrupt the model, it sends a turn_complete.For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded much more audio than has played yet.
            # 模型本身是支持被打断，即上一个回答还在持续输出的时候，提出新问题，模型会中止上个提问的输出，开始新问题的输出；
            # 所以，self.audio_queue队列中缓存的之前的回答需要清除，否则，就会将缓存中所有内容都播放
            # 因为是模型被打断的时候，会是一个新的输出，所以，在async for循环外面,不断重复的取出(移出)最后一个数据,直到队列为空
            # 潜在的问题也是:当for response in turn循环完毕,播放没有完成的话,后面的数据会被自动empty掉
            while not self.audio_out_queue.empty():  # 当加上这句块，有掉字的情况
                self.audio_out_queue.get_nowait()

    async def async_play_audio(
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

        while True:
            if self.pause_stream and not self.stop_stream:
                await asyncio.sleep(.1)  # 避免运行因长循环，滞留在此处，导致user_command阻塞
                continue
            elif not self.pause_stream and not self.stop_stream:
                audio_data = await self.audio_out_queue.get()  # asyncio.Queue是一个异步操作,需要await
                if audio_data is None:
                    self.stop_stream = True
                    stream.stop_stream()
                    stream.close()
                    self.logger.info('音频播放结束')
                    break
                await asyncio.to_thread(stream.write, audio_data)
            elif self.stop_stream:
                stream.stop_stream()
                stream.close()
                self.logger.info('user_command终止')
                break
    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_out_queue.get()
            await asyncio.to_thread(stream.write, bytestream)


if __name__ == "__main__":
    # asyncio.run(txt2txt())
    audioloop_instance = GeminiLiveStream()
    asyncio.run(audioloop_instance.run())
