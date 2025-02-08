import pyaudio
from pydub import AudioSegment

from async_pyaudio_record_player import create_wav_header

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

# 依赖: torchaudio, librosa, MinDAEC; 传送的音频格式可以接受: wav文件路径; pydub AudioSegment读取的内容; 也可以是音频bytes添加wave header后的字节串; 模型输入音频接受的最小长度是640ms
aec = pipeline(Tasks.acoustic_echo_cancellation, model="damo/speech_dfsmn_aec_psm_16k")
# output 为输出wav文件路径,如果不想输出文件,可以不列出output参数;但请不要将该参数设成output=None,否则会文件名None类型错误;
# result = aec(input={"nearend_mic":wav_data or wave file,"farend_speech":wave_data or wave_file}, output= wave_file_path)



if __name__ == "__main__":
    # 以下为aec模型测试输入的格式,经测试,模型输入可以接受: wav文件路径, pydub读取的内容, 也可以是音频bytes添加wave header后的字节串
    pya = pyaudio.PyAudio()

    input = {
        'nearend_mic': 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/nearend_mic.wav',
        'farend_speech': 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/farend_speech.wav'
    }

    nearend_mic_path = f"C:/Users/shoub/Music/nearend_mic1.wav"
    farend_speech_path = f"C:/Users/shoub/Music/farend_speech1.wav"
    nearend_mic_audio = AudioSegment.from_file(nearend_mic_path, format="wav")
    farend_speech_audio = AudioSegment.from_file(farend_speech_path, format="wav")
    print(f"nearend_mic_audio 类型: {type(nearend_mic_audio)}")
    # print(f"farend_speech_audio[:20]: {farend_speech_audio[:20]}")

    print(f"nearend_mic_audio.raw_data 类型: {type(nearend_mic_audio.raw_data)}")
    # print(f"farend_speech_audio.raw_data[:20]: {farend_speech_audio.raw_data[:20]}")

    nearend_bytes = nearend_mic_audio.raw_data
    farend_bytes = farend_speech_audio.raw_data

    nearend_bytes_wavhead = create_wav_header(nearend_bytes,16000,1,16)
    farend_bytes_wavhead = create_wav_header(farend_bytes,16000,1,16)



    stream = pya.open(
        format=pya.get_format_from_width(2),
        channels=1,
        rate=16000,
        output=True,
    )
    # stream.write(nearend_bytes)

    input = {
        'nearend_mic': nearend_bytes_wavhead,
        'farend_speech': farend_bytes_wavhead,
    }
    #
    result = aec(input, )
    stream.write(result['output_pcm'])