import asyncio
import pyaudio  # pyaudio 是一个跨平台的音频输入/输出库，主要用于处理 WAV 格式的音频数据
from pydub import AudioSegment  # pydub 库本身不直接播放音频文件，但它可以将多种格式的音频文件转换为 WAV 格式
import traceback
import logging


class Pyaudio_Record_Player:
    def __init__(
        self, pyaudio_instance: pyaudio.PyAudio, logger: logging.Logger = None
    ):
        self.pyaudio_instance = pyaudio_instance
        self.audio_queue = asyncio.Queue()
        self.pause_stream = False
        self.stop_stream = False

        if not logger:
            self.logger = logging.getLogger("Pyaudio_Record_Player")
            self.logger.setLevel("INFO")

    # 生成器函数，异步读取指定路径的音频文件
    async def user_command(self):
        """
        控制音频播放: pause or p for 暂停; stop or q or quit for 停止;
        """
        while True:
            user_input = await asyncio.to_thread(
                input, "pause/p: 暂停; stop/q/quit: 停止; 'c/continue: 继续: "
            )
            if user_input in ["pause", "p"]:
                self.pause_stream = True
                self.stop_stream = False
                self.logger.info(f"User input:{user_input}")
            elif user_input in ["c", "continue"]:
                self.pause_stream = False
                self.stop_stream = False
                self.logger.info(f"User input:{user_input}")
            elif user_input in ["q", "quit", "stop", "exit"]:
                self.pause_stream=False
                self.stop_stream = True
                self.logger.info(f"User input:{user_input}")
                break


    async def audiofile_read(self, file_path: str, chunk_size: int = 1024):
        """
        1. 非wav格式文件,音频转成wav;
        2. 输出byte类型raw data
        """
        # 使用 pydub 将音频文件转换为 WAV 格式
        audio = AudioSegment.from_file(file_path)
        channel = audio.channels
        sample_rate = audio.frame_rate
        self.logger.info(f"音频文件信息: 通道数:{channel},采样率:{sample_rate}")
        # 计算帧数
        n_frames = len(audio)
        for i in range(0, n_frames, chunk_size):  # 假设每次读取1024帧
            # 获取音频片段
            chunk = audio[i : i + chunk_size]
            # 将音频片段转换为字节
            data = chunk.raw_data
            await self.audio_queue.put(data)  # 写入Queue
            # yield data  # 输出音频内容

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
                continue
            elif not self.pause_stream and not self.stop_stream:
                audio_data = (
                    await self.audio_queue.get()
                )  # asyncio.Queue是一个异步操作,需要await
                # print(type(audio_data))
                await asyncio.to_thread(stream.write, audio_data)
            elif self.stop_stream:
                stream.stop_stream()
                stream.close()
                # self.pyaudio_instance.terminate()

    async def microphone_read(
        self,
        sample_width: int = 2,
        channels: int = 1,
        rate: int = 44100,
        chunk_size: int = 1024,
    ):
        """
        持续不断的从麦克风读取音频数据
        """
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

        while True:
            if self.pause_stream:
                continue
            elif not self.stop_stream:
                data = await asyncio.to_thread(audio_stream.read, chunk_size, **kwargs)
                # await self.audio_queue.put({"data": data, "mime_type": "audio/pcm"})
                await self.audio_queue.put(data)
                # yield data
            elif self.stop_stream:
                audio_stream.stop_stream()
                audio_stream.close()
                # self.pyaudio_instance.terminate()

    async def audiofile_player(
        self,
        file_path: str,
        sample_width: int = 2,
        channels: int = 2,
        rate: int = 44100,
        chunk_size: int = 1024,
    ):
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.audiofile_read(file_path, chunk_size))
                user_command_task = tg.create_task(self.user_command())
                tg.create_task(self.async_play_audio(sample_width, channels, rate))

                await user_command_task
                raise asyncio.CancelledError("停止播放")
        except asyncio.CancelledError:
            self.logger.info("停止播放")

        except ExceptionGroup as EG:
            self.pyaudio_instance.terminate()
            traceback.print_exception(EG)


# async def main(file_path: str, chunk_size: int = 1024):
#     """
#     :param file_path: 音频文件路径
#     :param chunk_size: 每次读取的音频数据大小
#     :return:
#     """
#     gen = audio_generator(file_path, chunk_size)
#
#     player_task = asyncio.create_task(async_play_audio(gen))
#
#     i = 0
#     while not player_task.done():
#         print(f"Main program: Playing audio chunk {i}")
#         await asyncio.sleep(1)
#         user_input = await asyncio.to_thread(
#             input, "为测试多任务,这里请输入:"
#         )  # input函数避免阻塞主程序,to_thread函数将input函数转换为单独线程
#         if user_input.lower() == "exit":
#             print("Exiting...,终端音乐播放,中止程序")
#             player_task.cancel()  # 关闭主程序之外的异步线程
#             break  # 退出主程序线程
#         else:
#             print(f"音乐播放不停止,你输入内容: {user_input}")
#         i += 1
#
#     # 主程序等待player_task完成,再结束
#     try:
#         await player_task
#     except asyncio.CancelledError:
#         print("Player task was cancelled.")
#
#
# async def record_play():
#     try:
#         # audio_in = await asyncio.to_thread(listen_audio)
#         # print(audio_in)
#         audio_in = listen_audio()
#         # async for data in audio_in:
#         #     print(data)
#         await asyncio.sleep(2)
#         async with asyncio.TaskGroup() as tg:
#             tg.create_task(async_play_audio(audio_in))
#             user_input = await asyncio.to_thread(input, "输入exit,中止录音播放:")
#             if user_input == "exit":
#                 raise asyncio.CancelledError("user requested exit")
#
#     except asyncio.CancelledError:
#         pass
#     except ExceptionGroup as EG:
#         traceback.print_exception(EG)


if __name__ == "__main__":
    pya = pyaudio.PyAudio()
    file_path = r"F:/Music/colorful days.mp3"
    player = Pyaudio_Record_Player(
        pya,
    )
    asyncio.run(player.audiofile_player(file_path))
