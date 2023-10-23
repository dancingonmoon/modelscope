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
output_dir = './result'
inference_pipeline_vad = pipeline(
    task=Tasks.voice_activity_detection,
    model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    model_revision=None,
    output_dir=output_dir,
    batch_size=1,
    mode='online',
)
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online',
    model_revision=None,
    # update_model=False,
    mode="paraformer_streaming",
    output_dir=output_dir,
)
inference_pipeline_punc = pipeline(
    task=Tasks.punctuation,
    model='damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727',
    model_revision=None,
)


# model_dir = os.path.join(os.environ["MODELSCOPE_CACHE"], "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")
# audio_data, sample_rate = soundfile.read(os.path.join(model_dir, "example/asr_example.wav"))

def RUN(audio_stream, speech_txt):
    """
    speech_txt: 之前累计的识别文本
    """
    samplerate = 16000  # 初始赋值,避免警告;
    speech = np.empty(None, dtype=np.float32)
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
        speech, samplerate = librosa.load(audio_stream, sr=16000, mono=True, offset=0.0, duration=None,
                                          dtype=np.float32)

    speech_length = speech.shape[0]

    # speech_stride_vad_segments = inference_pipeline_vad(audio_in=speech)

    sample_offset = 0
    chunk_size = [5, 10, 5]  # 第一个5为左看5帧,10为text_n 10帧,为600ms, 第二个5为右看5帧
    # stride_size = chunk_size[1] * 960 # 为什么是960 ?
    stride_size = chunk_size[1] * 960
    param_dict = {"cache": dict(), "is_final": False, "chunk_size": chunk_size}
    final_result = ""

    for sample_offset in range(0, speech_length, min(stride_size, speech_length - sample_offset)):
        if sample_offset + stride_size >= speech_length - 1:
            stride_size = speech_length - sample_offset
            param_dict["is_final"] = True
        else:
            param_dict["is_final"] = False

        speech_stride = speech[sample_offset:sample_offset + stride_size]
        # 1. VAD:
        speech_stride_vad_segments = inference_pipeline_vad(audio_in=speech_stride,
                                                            param_dict=param_dict)
        print(f"vad_segments:{speech_stride_vad_segments}")
        # 2. ASR:
        if 'text' in speech_stride_vad_segments:  # 检测到音频数据后，每隔600ms进行一次流式模型推理
            for i, segments in enumerate(speech_stride_vad_segments['text']):
                beg_idx = segments[0] * samplerate / 1000
                end_idx = segments[1] * samplerate / 1000
                speech_stride_vad_clip = speech_stride[int(beg_idx):int(end_idx)]
                # speech_stride_vad_clip = speech[int(beg_idx):int(end_idx)] # 取speech会导致错误
                speech_stride_vad_clip_result = inference_pipeline(audio_in=speech_stride_vad_clip,
                                                                   param_dict=param_dict)
                print(f"vad_{i}: {beg_idx}->{end_idx}: stride_vad_clip_result:{speech_stride_vad_clip_result}")

                # 3. punc
                # if speech_stride_vad_clip_result != []: # 例子中的代码,应该是错误的
                # (https://github.com/alibaba-damo-academy/FunASR/discussions/278)
                if len(speech_stride_vad_clip_result) != 0:  # 语音识别出非空时,
                    speech_stride_vad_clip_punc_result = inference_pipeline_punc(
                        text_in=speech_stride_vad_clip_result['text'],
                        param_dict=param_dict)
                    print(
                        f"punc_{i}: {beg_idx}->{end_idx}: stride_vad_clip_punc_result:{speech_stride_vad_clip_punc_result}")
                #
                else:
                    speech_stride_vad_clip_punc_result = speech_stride_vad_clip_result
                # 4. 拼接
                if len(speech_stride_vad_clip_punc_result) != 0:
                    final_result += speech_stride_vad_clip_punc_result['text']
        # else: # 当在说话停顿处，做标点断句恢复，修正识别文字:

        # if len(rec_result) != 0:
        #     final_result += rec_result['text']
        #     print(rec_result)

    speech_txt += final_result
    return speech_txt, speech_txt


with gr.Blocks(theme='soft', title='实时识别') as demo:
    with gr.Row(variant='panel'):
        audio_stream = gr.Audio(source='microphone', type='filepath', label='请录音并实时说话', streaming=True, )
        speech_txt = gr.State("")
        out = gr.Textbox(lines=2, placeholder='实时识别....', show_copy_button=True)

        audio_stream.stream(RUN, [audio_stream, speech_txt], [out, speech_txt])

demo.launch(debug=True, show_error=True, )
