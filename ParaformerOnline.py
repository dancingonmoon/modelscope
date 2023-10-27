import gradio as gr
import numpy as np

# import os
import logging

# import soundfile
import librosa

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)

# os.environ["MODELSCOPE_CACHE"] = "./"
output_dir = "./result"
inference_pipeline_vad = pipeline(
    task=Tasks.voice_activity_detection,
    model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    model_revision=None,
    output_dir=output_dir,
    batch_size=1,
    # mode="online",
)
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    model_revision=None,
    # update_model=False,
    mode="paraformer_streaming",
    output_dir=output_dir,
)
inference_pipeline_punc = pipeline(
    task=Tasks.punctuation,
    model="damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
    model_revision=None,
)

stride = 960 * 10 * 1


# model_dir = os.path.join(os.environ["MODELSCOPE_CACHE"], "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")
# audio_data, sample_rate = soundfile.read(os.path.join(model_dir, "example/asr_example.wav"))


def Stream_VAD_Stream(audio_stream, vad_stream, vad_flag=True):
    """
    audio_stream : 为每次输入的stream 片段;或为numpy,或为str
    vad_stream: 为每次输出的片段,初始为全0矩阵; 去除静音后的vad_stream累积长度, 因为每单一总是小于600ms, 长度大大小于9600;
    所以需要多个vad_stream拼接成更长的音频.
    vad_flag: 可以放弃vad,即,每个stream片段未经处理直接输出
    1. 对stream输入的整个片段进行VAD(offline);stream片段一般长度在24000-38000;
    2. 拼接vad之后的音频.刨除了静音部分;
    3. 输出vad_stream; numpy
    """
    samplerate = 16000  # 初始赋值,避免警告;
    speech = np.zeros(shape=(1,), dtype=np.float32)
    # 当输入是gr.Audio(type="Numpy")
    if isinstance(audio_stream, np.ndarray):
        samplerate, speech = audio_stream
        speech = speech.astype(np.float32)
        # gr.Audio转换的为(sample rate in Hz, audio data as a 16-bit int array);
        # Paraformer模型的音频输入要求: 1) 单声道;2) 16K采样率, 3) float32 缺一不可;
        # 双声道变成单声道
        if len(speech.shape) > 1:  # 说明是双声道
            speech = np.mean(speech, axis=1)
        # 采样率调整成16K:
        target_sr = 16000
        if samplerate != target_sr:
            speech = librosa.resample(speech, orig_sr=samplerate, target_sr=target_sr)

    if isinstance(audio_stream, str):
        # librosa.load 就可以解决采样率,单声道,float32
        speech, samplerate = librosa.load(
            audio_stream,
            sr=16000,
            mono=True,
            offset=0.0,
            duration=None,
            dtype=np.float32,
        )

    segments_list = []
    vad_len = 0
    if vad_flag:
        segments_list.append(vad_stream)
        print(f"vad合并之前的speech.shape:{speech.shape};拼接的vad_stream.shape:{vad_stream.shape}")
        segments_result = inference_pipeline_vad(audio_in=speech)
        print(f"vad之后的segments:{segments_result}")
        if "text" in segments_result:
            for segment in segments_result["text"]:
                segment_stream = speech[segment[0]: segment[1]]  # 当出现起始,终止为-1时,不知如何处理?
                segment_len = segment[1] - segment[0]
                vad_len += segment_len
                segments_list.append(segment_stream)
            vad_stream = np.concatenate(segments_list, axis=-1)

            #  当累积到长度超过9600*1时, 抛弃历史,保留最新:
            if vad_stream.shape[0] >= stride + vad_len + 1:  # 长度增长是跳跃性的,
                vad_stream = vad_stream[stride + vad_len:]

    else:
        vad_stream = speech

    return vad_stream


def RUN(audio_stream, speech_txt, vad_stream, vad_flag=False):
    """
    audio_stream : stream 片段, VAD之前;
    speech_txt: 之前累计的识别文本punc后的列表
    vad_stream : stream
    """
    speech = Stream_VAD_Stream(audio_stream, vad_stream, vad_flag=vad_flag)
    if np.all(speech == 0) or (vad_flag and speech.shape[0] < stride):  # 排除全0 ,无内容音频;排除长度不到stride,不予识别
        return speech_txt, speech_txt, speech
    else:
        print(f"vad合并之后的speech.shape:{speech.shape}")
        speech_length = speech.shape[0]
        print(
            f"后台收到的speech.shape: {speech.shape};speech_length: {speech_length};每个stride步长:9600"
        )
        sample_offset = 0
        chunk_size = [5, 10, 5]  # 第一个5为左看5帧,10为text_n 10帧,为600ms, 第二个5为右看5帧
        # stride_size = chunk_size[1] * 960 # 为什么是960 ?
        stride_size = chunk_size[1] * 960
        param_dict = {"cache": dict(), "is_final": False, "chunk_size": chunk_size}
        param_punc = {"cache": []}
        # final_result = ""
        # punc_list = []  # list以送入punc online mode
        # punc_list.append(speech_txt) # 将之前的punc后的txt,作为vad之后的第一个,再送入pineline_punc()

        for sample_offset in range(
                0, speech_length, min(stride_size, speech_length - sample_offset)
        ):
            print(f"sample: {sample_offset} of {speech_length}")
            if sample_offset + stride_size >= speech_length - 1:
                stride_size = speech_length - sample_offset
                param_dict["is_final"] = True
            else:
                param_dict["is_final"] = False

            speech_stride = speech[sample_offset: sample_offset + stride_size]
            # 1. VAD 输入时,已经完成;
            # 2. ASR:
            speech_stride_result = inference_pipeline(
                audio_in=speech_stride, param_dict=param_dict
            )
            # 3. punc
            # if speech_stride_vad_clip_result != []: # 例子中的代码,应该是错误的
            # (https://github.com/alibaba-damo-academy/FunASR/discussions/278)
            # the online model is used at each time of speech absence of voice activity detection(VAD)
            # and should be feed the cache extracted from the history text.
            if "text" in speech_stride_result:  # 语音识别出非空时,
                speech_stride_punc_result = inference_pipeline_punc(
                    text_in=speech_stride_result["text"],
                    param_dict=param_punc,
                )
                print(f"stride_punc_result:{speech_stride_punc_result}")
            else:
                speech_stride_punc_result = speech_stride_result

            if "text" in speech_stride_punc_result:
                speech_txt += speech_stride_punc_result["text"]

    return speech_txt, speech_txt, speech


with gr.Blocks(theme="soft", title="实时识别") as demo:
    with gr.Row(variant="panel"):
        with gr.Column():
            audio_stream = gr.Audio(
                source="microphone",
                type="filepath",
                label="请录音并实时说话",
                streaming=True,
            )
            vad_flag = gr.Checkbox(value=False, label='VAD', show_label=True, )
        speech_txt = gr.State("")
        out = gr.Textbox(lines=2, placeholder="实时识别....", show_copy_button=True)
        vad_stream_var = gr.State(np.zeros(shape=(1,), dtype=np.float32))
        audio_stream.stream(
            RUN,
            [audio_stream, speech_txt, vad_stream_var, vad_flag],
            [out, speech_txt, vad_stream_var],
        )

demo.launch(
    debug=True,
    show_error=True,
)
