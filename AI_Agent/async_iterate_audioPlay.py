import asyncio
import pyaudio  # pyaudio 是一个跨平台的音频输入/输出库，主要用于处理 WAV 格式的音频数据
from pydub import AudioSegment  # pydub 库本身不直接播放音频文件，但它可以将多种格式的音频文件转换为 WAV 格式
import traceback

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 44100
RECEIVE_SAMPLE_RATE = 44100
CHUNK_SIZE = 1024

pya = pyaudio.PyAudio()
# audio_in_queue = asyncio.Queue()
# audio_out_queue = asyncio.Queue()

# 生成器函数，异步读取指定路径的音频文件
async def audio_generator(file_path:str, chunk_size:int=1024):
    # 使用 pydub 将音频文件转换为 WAV 格式
    audio = AudioSegment.from_file(file_path)
    channel = audio.channels
    sample_rate = audio.frame_rate
    print(f"音频文件信息: 通道数:{channel},采样率:{sample_rate}")
    # 计算帧数
    nframes = len(audio)
    for i in range(0, nframes, chunk_size):  # 假设每次读取1024帧
        # 获取音频片段
        chunk = audio[i:i+chunk_size]
        # 将音频片段转换为字节
        data = chunk.raw_data
        yield data  # 输出音频内容

async def async_play_audio(gen, channels=2, rate=44100):
    # p = pyaudio.PyAudio()
    stream = pya.open(format=pya.get_format_from_width(2),
                      channels=channels,
                      rate=rate,
                      output=True)

    async for audio_data in gen:
        # print(f"Playing audio chunk {i}")
        stream.write(audio_data)
        await asyncio.sleep(0)  # 让出控制权，使其他任务能够运行

    stream.stop_stream()
    stream.close()
    # p.terminate()

async def listen_audio():
    """
    持续不断的从麦克风读取音频数据
    """
    mic_info = pya.get_default_input_device_info()
    audio_stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_SAMPLE_RATE,
        input=True,
        input_device_index=mic_info["index"],
        frames_per_buffer=CHUNK_SIZE,
    )
    if __debug__:  # 这是一个条件语句，用于检查当前是否处于调试模式。
        kwargs = {"exception_on_overflow": False}  # 如果处于调试模式，kwargs 被设置为一个字典，这意味着在调试模式下，当音频流溢出（即缓冲区满，无法再接收更多数据）时，不会抛出异常。
    else:
        kwargs = {}   # 如果当前不是调试模式，这意味着在生产模式下，当音频流溢出时，可能会抛出异常，这通常是为了确保程序能够及时处理错误情况
    while True:
        data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, **kwargs)
        # yield {"data": data, "mime_type": "audio/pcm"}
        yield data
async def main(file_path:str, chunk_size:int=1024):
    """
    :param file_path: 音频文件路径
    :param chunk_size: 每次读取的音频数据大小
    :return:
    """
    gen = audio_generator(file_path,chunk_size)

    player_task = asyncio.create_task(async_play_audio(gen))

    i = 0
    while not player_task.done():
        print(f"Main program: Playing audio chunk {i}")
        await asyncio.sleep(1)
        user_input = await asyncio.to_thread(input, "为测试多任务,这里请输入:") # input函数避免阻塞主程序,to_thread函数将input函数转换为单独线程
        if user_input.lower() == "exit":
            print("Exiting...,终端音乐播放,中止程序")
            player_task.cancel()  # 关闭主程序之外的异步线程
            break # 退出主程序线程
        else:
            print(f"音乐播放不停止,你输入内容: {user_input}")
        i += 1


    # 主程序等待player_task完成,再结束
    try:
        await player_task
    except asyncio.CancelledError:
        print("Player task was cancelled.")

async def record_play():
    try:
        # audio_in = await asyncio.to_thread(listen_audio)
        # print(audio_in)
        audio_in = listen_audio()
        # async for data in audio_in:
        #     print(data)
        await asyncio.sleep(2)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(async_play_audio(audio_in))
            user_input =  await asyncio.to_thread(input, "输入exit,中止录音播放:")
            if user_input =='exit':
                raise asyncio.CancelledError("user requested exit")

    except asyncio.CancelledError:
        pass
    except ExceptionGroup as EG:
        traceback.print_exception(EG)







if __name__ == "__main__":
    # file_path = r"C:/Users/shoub/Music/相思.mp3"
    # asyncio.run(main(file_path))
    asyncio.run(record_play())
