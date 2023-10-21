import gradio as gr
import numpy as np

import os
import logging
import soundfile
import librosa

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)

# os.environ["MODELSCOPE_CACHE"] = "./"
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online',
    model_revision= None,
    # update_model=False,
    mode="paraformer_streaming"
)

# model_dir = os.path.join(os.environ["MODELSCOPE_CACHE"], "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")
# audio_data, sample_rate = soundfile.read(os.path.join(model_dir, "example/asr_example.wav"))

def RUN(audio_stream,):
    """

    """
    speech_length = audio_stream.shape[0]
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

    # result = inference_pipeline(speech)

    sample_offset = 0
    chunk_size = [5, 10, 5] #第一个5为左看5帧,10为text_n 10帧,为600ms, 第二个5为右看5帧
    stride_size =  chunk_size[1] * 960
    param_dict = {"cache": dict(), "is_final": False, "chunk_size": chunk_size}
    final_result = ""

    for sample_offset in range(0, speech_length, min(stride_size, speech_length - sample_offset)):
        if sample_offset + stride_size >= speech_length - 1:
            stride_size = speech_length - sample_offset
            param_dict["is_final"] = True
        rec_result = inference_pipeline(audio_in=speech[sample_offset: sample_offset + stride_size],
                                    param_dict=param_dict)
        if len(rec_result) != 0:
            final_result += rec_result['text']
            print(rec_result)

    return final_result

# def transcribe(stream, new_chunk):
#     sr, y = new_chunk
#     y = y.astype(np.float32)
#     y /= np.max(np.abs(y))
#
#     if stream is not None:
#         stream = np.concatenate([stream, y])
#     else:
#         stream = y
#     return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]


demo = gr.Interface(
    RUN,
    gr.Audio(source="microphone", streaming=True),
    gr.Textbox(),
    live=True,
)

demo.launch()
