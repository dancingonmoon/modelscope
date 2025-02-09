import asyncio
from collections import deque

import pyaudio  # pyaudio 是一个跨平台的音频输入/输出库，主要用于处理 WAV 格式的音频数据
from pydub import AudioSegment  # pydub 库本身不直接播放音频文件，但它可以将多种格式的音频文件转换为 WAV 格式
import traceback
import logging
import webrtcvad
import numpy as np
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        # logging.FileHandler("app.log")  # 输出到文件
    ],
)

# 阿里通义实验室发布的DFSMN回声消除模型：一种音频通话场景的单通道回声消除模型算法
# (https://modelscope.cn/models/iic/speech_dfsmn_aec_psm_16k/summary)
# 模型pipeline 输入为两个16KHz采样率的单声道wav文件，分别是本地麦克风录制信号和远端参考信号，输出结果保存在指定的wav文件中
# 模型局限性:1)由于训练数据偏差，如果麦克风通道存在音乐声，则音乐会被抑制。2)麦克风和参考通道之间的延迟覆盖范围在500ms以内
# 阿里的Audio语音模型，通常建议安装audio领域依赖：pip install "modelscope[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
# 因部分依赖由ModelScope独立host，所以需要使用"-f"参数
# 譬如在windows系统安装，可能会遇到缺少依赖： MinDAEC, 安装该依赖需要在modelscope的独立host里面寻找，需要添加参数：
# pip install "MinDAEC" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

aec_enabled = True  # 是否启用回声消除;启动aec 需要启动speech_dfsmn_aec_psm_16k模型，该模型需要依赖torchaudio, librosa, MinDAEC
if aec_enabled:
    # 依赖: torchaudio, librosa, MinDAEC; 传送的音频格式可以接受: wav文件路径; pydub AudioSegment读取的内容; 也可以是音频bytes添加wave header后的字节串; 模型输入音频接受的最小长度是640ms
    aec = pipeline(Tasks.acoustic_echo_cancellation, model="damo/speech_dfsmn_aec_psm_16k")
    # output 为输出wav文件路径,如果不想输出文件,可以不列出output参数;但请不要将该参数设成output=None,否则会文件名None类型错误;
    # result = aec(input={"nearend_mic":wav_data or wave file,"farend_speech":wave_data or wave_file}, output= wave_file_path)

def create_wav_header(dataflow, sample_rate=16000, num_channels=1, bits_per_sample=16):
    """
    创建WAV文件头的字节串。 (替代生成wave文件）

    :param dataflow: 音频bytes数据（以字节为单位）。
    :param sample_rate: 采样率，默认16000。
    :param num_channels: 声道数，默认1（单声道）。
    :param bits_per_sample: 每个样本的位数，默认16。
    :return: WAV文件头的字节串和音频bytes数据。
    """
    total_data_len = len(dataflow)
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_chunk_size = total_data_len
    fmt_chunk_size = 16
    riff_chunk_size = 4 + (8 + fmt_chunk_size) + (8 + data_chunk_size)

    # 使用 bytearray 构建字节串
    header = bytearray()

    # RIFF/WAVE header
    header.extend(b'RIFF')
    header.extend(riff_chunk_size.to_bytes(4, byteorder='little'))
    header.extend(b'WAVE')

    # fmt subchunk
    header.extend(b'fmt ')
    header.extend(fmt_chunk_size.to_bytes(4, byteorder='little'))
    header.extend((1).to_bytes(2, byteorder='little'))  # Audio format (1 is PCM)
    header.extend(num_channels.to_bytes(2, byteorder='little'))
    header.extend(sample_rate.to_bytes(4, byteorder='little'))
    header.extend(byte_rate.to_bytes(4, byteorder='little'))
    header.extend(block_align.to_bytes(2, byteorder='little'))
    header.extend(bits_per_sample.to_bytes(2, byteorder='little'))

    # data subchunk
    header.extend(b'data')
    header.extend(data_chunk_size.to_bytes(4, byteorder='little'))

    return bytes(header) + dataflow

class Pyaudio_Record_Player:
    def __init__(
        self, pyaudio_instance: pyaudio.PyAudio, logger: logging.Logger = None
    ):
        self.pyaudio_instance = pyaudio_instance
        self.audio_queue = asyncio.Queue()  # 音频缓冲器
        self.audio_out = None  # 存放输出音频,以作为回声抑制算法中的参考信号
        self.audio_out_deque = deque(maxlen=10)  # 存放输出音频以及对应的时间戳队列，以对齐麦克风录音输入，考虑到最大延时，设定10*640ms=6.4s延迟
        self.pause_stream = False
        # self.stop_stream = False
        self.stop_stream = asyncio.Event()  # 创建等待事件,来控制Taskgroup()
        self.audio_play_channels = None  # 用于音频文件播放，从文件中提取
        self.audio_play_sample_rate = None  # 用于音频文件播放，从文件中提取
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
                    # self.stop_stream = False
                    self.stop_stream.clear()
                    self.logger.info(f"User input: {user_input}")
                elif user_input in ["c", "continue"]:
                    self.pause_stream = False
                    # self.stop_stream = False
                    self.stop_stream.clear()
                    self.logger.info(f"User input: {user_input}")
                elif user_input in ["q", "quit", "stop", "exit"]:
                    self.pause_stream = False
                    self.stop_stream.set()
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
        audio = AudioSegment.from_file(file_path)
        self.audio_play_channels = audio.channels
        self.audio_play_sample_rate = audio.frame_rate
        self.logger.info(
            f"音频文件信息: 通道数:{self.audio_play_channels},采样率:{self.audio_play_sample_rate}"
        )
        # 计算帧数
        n_frames = len(audio)
        for i in range(0, n_frames, chunk_size):  # 假设每次读取1024ms
            # 获取音频片段
            chunk = audio[i : i + chunk_size]
            # 将音频片段转换为字节
            data = chunk.raw_data
            await self.audio_queue.put(data)  # 写入Queue
            if self.stop_stream.is_set():
                break
        await self.audio_queue.put(None)  # signal the end of the audio

    async def async_audio_play(
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

        while not self.stop_stream.is_set():
            if self.pause_stream:
                await asyncio.sleep(0.1)  # 避免运行因长循环，滞留在此处，导致user_command阻塞
                continue
            self.audio_out, vad_mark = await self.audio_queue.get()  # asyncio.Queue是一个异步操作,需要await ;
            if self.audio_out is None:
                self.stop_stream.set()
                self.logger.info("音频播放结束")
                break

            await asyncio.to_thread(stream.write, self.audio_out)
            self.audio_out_deque.append((time.time(), self.audio_out, vad_mark))  # 添加到deque队列,队列存放延迟累计chunk的播放音频,并带每个chunk播放时间的时间戳
        if self.stop_stream.is_set():
            stream.stop_stream()
            stream.close()
            self.logger.info("stop and close stream")

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
        不处理回声抑制, 但有VAD
        1. 经过vad判断is_speech,静音填充
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
            # 准备静音数据,为稍后使用
            silent_frame = np.zeros(chunk_size, dtype=np.int16).tobytes()

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
                    is_speech = vad.is_speech(data_np, rate, length=int(chunk_size/2))  # 传入的音频数据是未压缩的 PCM 数据，并且数据类型是 int16
                    # self.logger.info(f"vad: is_speech={is_speech}")
                    if is_speech:
                        audio_vad = data
                    else:
                        # 填补静音数据
                        audio_vad = silent_frame

                    await self.audio_queue.put((audio_vad, is_speech))  # 音频缓冲器除了放置了音频数据,还放置了该音频数据是否是vad的结果,is_speech的bool值

                except OSError as e:
                    self.logger.error(f"麦克风读取发生操作系统错误: {e}")
                    break  # 发生错误时退出循环
            if self.stop_stream.is_set():
                audio_stream.stop_stream()
                audio_stream.close()
                self.logger.info("等待事件set,或操作系统错误,停止关闭stream")

        except ExceptionGroup as EG:
            traceback.print_exception(EG)
            self.logger.info(f"麦克风初始化或读取发生错误: {EG}")
    async def microphone_read_AEC(
        self,
        sample_width: int = 2,
        channels: int = 1,
        rate: int = 16000,
        chunk_size: int = 480,  # 16Khz, 30ms长度,对应的帧长度是480
        vad_mode: int = 2,  # vad 模式，0-3，3最敏感
    ):
        """
        持续不断的从麦克风读取音频数据;使用asyncio.Queue来缓存队列,传递异步进程的音频数据,音频输入输出更加光滑;
        实现了回声抑制:
        1. 经过vad判断is_speech,静音填充
        2. 使用阿里通义实验室DFSMN回声消除模型:模型接受单通道麦克风信号和单通道参考信号作为输入，输出线性回声消除和回声残余抑制后的音频信号
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

            present_chunk_time = time.time() # 初始化第一个chunk的时间戳
            # 准备静音数据,为稍后使用
            silent_frame = np.zeros(chunk_size, dtype=np.int16).tobytes()

            while not self.stop_stream.is_set():
                if self.pause_stream:
                    await asyncio.sleep(0.1)
                    continue
                try:
                    data = await asyncio.to_thread(
                        audio_stream.read, chunk_size, **kwargs
                    )
                    # self.logger.info(f"麦克风data type:{type(data)}")
                    # self.logger.info(f"麦克风data 内容:{data[:50]}")
                    data_np = np.frombuffer(data, dtype=np.int16)

                    # 使用VAD检测是否是语音
                    is_speech = vad.is_speech(data_np, rate, length=int(chunk_size/2))  # 传入的音频数据是未压缩的 PCM 数据，并且数据类型是 int16
                    self.logger.info(f"vad: is_speech={is_speech}")
                    if is_speech:
                        audio_vad = data
                    else:
                        # 填补静音数据
                        audio_vad = silent_frame
                        # 麦克风无voice输入,则直接将audio_vad 送入self.audio_queue(缓冲器);跳过aec模型处理
                        # 音频缓冲器存放了元组(音频数据,vad结果is_speech的bool值)
                        await self.audio_queue.put((audio_vad, is_speech))
                        # present_chunk_time = time.time()
                        continue

                    #  这里将添加对self.audio_out_deque里面的每个(时间戳,以及self.audio_out)的处理,添加wave_head,再与录音的chunk比较时间戳.匹配一个chunk时长以内的录音chunk与播放chunk
                    #  1) 获取录音每个chunk生成的时间戳,并与self.audio_out_deque里面的时间戳比较,取最接近的元素(元素),并且时间戳的延迟少于chunk时长,以匹配录音chunk与播放chunk;
                    #  2) 将匹配的播放chunk的self.audio_out添加wavehead,送入self.audio_queue(缓冲器)
                    #  3) 如果没有找到匹配的录音与播放chunk,则表明某种原因,例如播放尚未开始,不存在录音与播放chunk的匹配对,则跳过aec模型处理,直接将录音chunk送入self.audio_queue(缓冲器)
                    matched_audio = None
                    matched_timedelay = float('inf')  # 初始化匹配chunk的时延
                    if self.audio_out_deque:  # deque可能初始为空
                        # self.logger.info(f"self.audio_out_deque:{[timestamp for timestamp, _, _ in self.audio_out_deque]}")
                        for timestamp, audio_out, vad_mark in self.audio_out_deque:
                            timedelay = abs(timestamp - present_chunk_time)
                            if timedelay <= chunk_size/1000 and timedelay < matched_timedelay and not vad_mark:
                                matched_timedelay = timedelay
                                matched_audio = audio_out


                    if matched_audio:
                        # 将bytes音频转换成wav格式 (创建wav头的字节串）
                        nearend_mic_bytes = create_wav_header(audio_vad, rate, channels,
                                                              bits_per_sample=sample_width * 8)
                        farend_speech_bytes = create_wav_header(matched_audio, rate, channels,
                                                                bits_per_sample=sample_width * 8)
                        audio_echo_cancellation = aec(input={'nearend_mic': nearend_mic_bytes,
                                                             'farend_speech': farend_speech_bytes},
                                                      )  # aec输出为一个字典{'output_pcm': b'\x00\x00‘} ;
                        await self.audio_queue.put((audio_echo_cancellation['output_pcm'], is_speech))
                        present_chunk_time = time.time()
                        self.logger.info(f"找到匹配的录音与播放chunk,录音chunk与播放chunk的时延:{matched_timedelay}, vad_mark={vad_mark}")
                        self.logger.info(f"nearend_mic_bytes:{nearend_mic_bytes[-50:]}")
                        self.logger.info(f"farend_speh_bytes:{farend_speech_bytes[-50:]}")
                        self.logger.info(f"nearend wavehead之前 长度:{len(audio_vad)}")
                        # self.logger.info(f"nearend wavehead之后 长度:{len(nearend_mic_bytes)}")
                        self.logger.info(f"farend wavehead之前 长度:{len(matched_audio)}")
                        # self.logger.info(f"farend wavehead之后 长度:{len(farend_speech_bytes)}")
                        self.logger.info(f"aec之后的output_pcs[:50]:{audio_echo_cancellation['output_pcm'][:50]}")
                        self.logger.info(f"aec之后的output_pcs[-50:]:{audio_echo_cancellation['output_pcm'][-50:]}")
                        self.logger.info(f"aec之后的output_pcs 长度:{len(audio_echo_cancellation['output_pcm'])}")
                        continue

                    else:# 未找到匹配项,跳过aec,直接将录音chunk送入self.audio_queue
                        await self.audio_queue.put((audio_vad, is_speech))
                        present_chunk_time = time.time()


                except OSError as e:
                    self.logger.error(f"麦克风读取发生操作系统错误: {e}")
                    break  # 发生错误时退出循环
            if self.stop_stream.is_set():
                audio_stream.stop_stream()
                audio_stream.close()
                self.logger.info("等待事件set,或操作系统错误,停止关闭stream")

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
                user_command = tg.create_task(self.user_command())
                await asyncio.sleep(
                    0.1
                )  # 等待self.channels, self.sample_rate被赋值完成;否则会导致self.channels=None, self.sample_rate=None
                tg.create_task(
                    self.async_audio_play(
                        sample_width,
                        self.audio_play_channels,
                        self.audio_play_sample_rate,
                    )
                )
                # 以下代码由于user_command异步任务在quit时,自会break,而取消tg的所有任务,并无必要;
                # TaskGroup的目的是管理一组相互依赖的任务，这些任务应该一起启动和结束。
                # 当一个任务完成时，TaskGroup认为整个任务组的工作已经完成，因此会尝试取消其他所有任务
                # await self.stop_stream.wait()
                # self.logger.info("stop_stream等待事件阻塞被解除")

        except ExceptionGroup as EG:
            traceback.print_exception(EG)
        finally:
            self.pyaudio_instance.terminate()  # 在程序结束时调用一次
            self.logger.info("播放器已清理资源")

    async def microphone_test(
        self,
        sample_width: int = 2,
        channels: int = 1,
        rate: int = 16000,
        chunk_size: int = 480,  # 16Khz, 30ms长度,对应的帧长度是480
        aec_enabled: bool = False
    ):
        """
        麦克风收音,3秒后回放;user_command控制;
        """
        try:
            async with asyncio.TaskGroup() as tg:
                if aec_enabled:
                    tg.create_task(
                    self.microphone_read_AEC(sample_width, channels, rate, chunk_size)
                    )
                else:
                    tg.create_task(
                        self.microphone_read(sample_width, channels, rate, chunk_size)
                    )

                tg.create_task(self.user_command())
                await asyncio.sleep(3)
                tg.create_task(self.async_audio_play(sample_width, channels, rate))
                # TaskGroup的目的是管理一组相互依赖的任务，这些任务应该一起启动和结束。
                # 当一个任务完成时，TaskGroup认为整个任务组的工作已经完成，因此会尝试取消其他所有任务
                await self.stop_stream.wait()
                self.logger.info("stop_stream等待事件阻塞解除")
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
    player = Pyaudio_Record_Player(pya, logger)
    # asyncio.run(player.audiofile_player(file_path))
    asyncio.run(
        player.microphone_test(sample_width=2, channels=1, rate=16000, chunk_size=640, aec_enabled=aec_enabled)
    )

