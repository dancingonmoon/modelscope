import asyncio
import pyaudio  # pyaudio 是一个跨平台的音频输入/输出库，主要用于处理 WAV 格式的音频数据
from pydub import AudioSegment  # pydub 库本身不直接播放音频文件，但它可以将多种格式的音频文件转换为 WAV 格式
import traceback
import logging
import webrtcvad
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # 输出到控制台
                        # logging.FileHandler("app.log")  # 输出到文件
                    ])


class Pyaudio_Record_Player:
    def __init__(
            self, pyaudio_instance: pyaudio.PyAudio, logger: logging.Logger = None
    ):
        self.pyaudio_instance = pyaudio_instance
        self.audio_queue = asyncio.Queue()
        self.pause_stream = False
        self.stop_stream = False
        # self.playback_end = False
        self.channels = 2
        self.sample_rate = 44100
        self.logger = logger

        if not logger:
            self.logger = logging.getLogger("Pyaudio_Record_Player")
            self.logger.setLevel("INFO")

    # 生成器函数，异步读取指定路径的音频文件
    async def user_command(self):
        """
        控制音频播放: pause/p: 暂停; stop/q/quit: 停止; 'c/continue: 继续;
        """
        try:
            while True:
                user_input = await asyncio.to_thread(
                    input, "pause/p: 暂停; c/continue: 继续; stop/q/quit: 停止: "
                )
                if user_input in ["pause", "p"]:
                    self.pause_stream = True
                    self.stop_stream = False
                    self.logger.info(f"User input: {user_input}")
                elif user_input in ["c", "continue"]:
                    self.pause_stream = False
                    self.stop_stream = False
                    self.logger.info(f"User input: {user_input}")
                elif user_input in ["q", "quit", "stop", "exit"]:
                    self.pause_stream = False
                    self.stop_stream = True
                    self.logger.info(f"User input: {user_input}")
                    break
                else:
                    self.logger.info(f"invalid User input: {user_input}")

                await asyncio.sleep(0.1)  # 避免运行阻塞在user_command,其它异步线程停滞
        # except ValueError:
        #     self.logger.info("Standard input stream closed, user command exiting.")
        # 存在问题未解： 当正常播放结束，asyncio.to_thread(input)异步线程等待键盘输入，程序无法关闭
        except asyncio.CancelledError:
            self.logger.info("user_command task cancelled.")

    async def audiofile_read(self, file_path: str, chunk_size: int = 1024):
        """
        1. 非wav格式文件,音频转成wav;
        2. 输出byte类型raw data
        """
        # 使用 pydub 将音频文件转换为 WAV 格式
        # self.playback_end = False
        audio = AudioSegment.from_file(file_path)
        self.channels = audio.channels
        self.sample_rate = audio.frame_rate
        self.logger.info(f"音频文件信息: 通道数:{self.channels},采样率:{self.sample_rate}")
        # 计算帧数
        n_frames = len(audio)
        for i in range(0, n_frames, chunk_size):  # 假设每次读取1024ms
            # 获取音频片段
            chunk = audio[i: i + chunk_size]
            # 将音频片段转换为字节
            data = chunk.raw_data
            await self.audio_queue.put(data)  # 写入Queue
            if self.stop_stream:
                break
        if not self.stop_stream:
            await self.audio_queue.put(None)  # signal the end of the audio

    async def async_play_audio(
            self,
            sample_width: int = 2,
            channels: int = 2,
            rate: int = 44100,
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
                audio_data = await self.audio_queue.get()  # asyncio.Queue是一个异步操作,需要await
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

    async def microphone_read(
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
            while True:
                if self.pause_stream:
                    await asyncio.sleep(.1)
                    continue
                elif not self.stop_stream:
                    try:
                        data = await asyncio.to_thread(audio_stream.read, chunk_size, **kwargs)
                        data_np = np.frombuffer(data, dtype=np.int16)
                        # 使用VAD检测是否是语音
                        is_speech = vad.is_speech(data_np.tobytes(), rate)
                        if is_speech:
                            audio_vad = data
                        else:
                            # 填补静音数据
                            silent_frame = np.zeros(chunk_size, dtype=np.int16)
                            audio_vad = silent_frame.tobytes()
                        await self.audio_queue.put(audio_vad)
                    except OSError as e:
                        self.logger.error(f"麦克风读取发生操作系统错误: {e}")
                        break  # 发生错误时退出循环
                elif self.stop_stream:
                    audio_stream.stop_stream()
                    audio_stream.close()
                    self.logger.info('麦克风停止录音')
                    break
        except ExceptionGroup as EG:
            traceback.print_exception(EG)
            self.logger.info(f"麦克风初始化或读取发生错误: {EG}")

    async def audiofile_player(
            self,
            file_path: str,
            sample_width: int = 2,
            chunk_size: int = 1024,
    ):
        """
        存在问题未解： 当正常播放结束，asyncio.to_thread(input)异步线程等待键盘输入，程序无法关闭
        正常播放结束,需要键盘手动输入stop/q/quit程序才会关闭.
        尽管async_play_audio在音频结束时,关闭了该函数,但是已经等待用户输入的input()无法由程序控制来关闭,
        """
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.audiofile_read(file_path, chunk_size))
                user_command_task = tg.create_task(self.user_command())
                tg.create_task(self.async_play_audio(sample_width, self.channels, self.sample_rate))
                # while not self.stop_stream:
                #     await asyncio.sleep(.1)
                # user_command_task.cancel(f'stop_stream:{self.stop_stream}')


        except ExceptionGroup as EG:
            traceback.print_exception(EG)
        finally:
            self.pyaudio_instance.terminate()  # 在程序结束时调用一次
            self.logger.info("播放器已清理资源")
            # sys.stdin.close()  # 当播放正常结束时，asyncio.to_thread(input)有个异步线程等待input

    async def microphone_test(self,
                              sample_width: int = 2,
                              channels: int = 1,
                              rate: int = 16000,
                              chunk_size: int = 480,  # 16Khz, 30ms长度,对应的帧长度是480
                              ):
        """
        麦克风收音,3秒后回放;user_command控制;
        """
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.microphone_read(sample_width, channels, rate, chunk_size))
                tg.create_task(self.user_command())
                await asyncio.sleep(3)
                tg.create_task(self.async_play_audio(sample_width, channels, rate))
        except ExceptionGroup as EG:
            traceback.print_exception(EG)
        finally:
            self.pyaudio_instance.terminate()  # 在程序结束时调用一次
            self.logger.info("pyaudio实例已清理资源")


if __name__ == "__main__":
    logger = logging.getLogger("Pyaudio_Record_Player")
    logger.setLevel("INFO")
    pya = pyaudio.PyAudio()
    file_path = r"F:/Music/放牛班的春天10.mp3"
    # file_path = r"H:/music/Music/color of the world.mp3"
    player = Pyaudio_Record_Player(
        pya,
    )
    # asyncio.run(player.audiofile_player(file_path))
    asyncio.run(player.microphone_test(sample_width=2, channels=1, rate=16000, chunk_size=480))
