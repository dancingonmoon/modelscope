from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import gradio as gr
import numpy as np

import librosa
import time

def ms2strftime(timestamp):
    """
    将毫秒值转换为可读的时间格式
    timestamp: 为毫秒值
    """
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(timestamp/1000))
    return formatted_time

def Paraformer_longaudio_model(
    use_vad_model=True, use_punc_model=True, use_lm_model=False
):

    if use_vad_model:
        vad_model = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    else:
        vad_model = ""

    if use_punc_model:
        punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
        # punc_model = "damo/punc_ct-transformer_cn-en-common-vocab471067-large"

    else:
        punc_model = ""

    output_dir = "./results"
    if use_lm_model:
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            # defaults to combine VAD, ASR and PUNC
            model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            # model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            # model="damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn",  # 分角色语音识别
            vad_model=vad_model,
            punc_model=punc_model,
            lm_model="damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch",
            lm_weight=0.15,
            beam_size=10,
            model_revision=None,
            # model_revision="v0.0.2",
            output_dir=output_dir,
        )

    else:
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            # defaults to combine VAD, ASR and PUNC
            model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            # model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            # model="damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn",  # 分角色语音识别
            vad_model=vad_model,
            punc_model=punc_model,
            model_revision=None,
            # model_revision="v0.0.2",
            output_dir=output_dir,
        )

    return inference_pipeline  # 这里先输出模型, 以避免后续模型重复生成;

inference_pipeline = Paraformer_longaudio_model() # 使用缺省值生成模型;

def RUN(audio_data, model_selected, 
        models_change_flag=False, use_timestamp=True):
    """
    audio_data: 为输入音频,为gr.Audio输出,为元组: (int sample rate, numpy.array for the data),二进制数据(bytes);url
    """
    global inference_pipeline 
    # 判断模型选择是否发生变化,并重新加载模型:
    if models_change_flag:        
        use_vad_model = True if "VAD" in model_selected else False
        use_punc_model = True if "PUNC" in model_selected else False
        use_lm_model = True if "NNLM" in model_selected else False
        
        inference_pipeline = Paraformer_longaudio_model(
            use_vad_model=use_vad_model, use_punc_model=use_punc_model, use_lm_model=use_lm_model)
    
    samplerate, waveform = audio_data
    waveform = waveform.astype(np.float32)  
    # gr.Audio转换的为(sample rate in Hz, audio data as a 16-bit int array);
    # Paraformer模型的音频输入要求: 1) 单声道;2) 16K采样率, 3) float32 缺一不可;
    # 双声道变成单声道
    if len(waveform.shape) > 1:  # 说明是双声道
        waveform = np.mean(waveform, axis=1)
    # 采样率调整成16K:
    target_sr = 16000
    if samplerate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=samplerate, target_sr=target_sr)
    
    result = inference_pipeline(waveform, use_timestamp=use_timestamp) 

    # 读出内容:
    contents = ""
    if use_timestamp:
        for dic in result["sentences"]:
            start = ms2strftime(dic['start'])
            end = ms2strftime(dic['end'])
            contents += f"{start}->{end} : {dic['text']} \n"
    else:
        contents = result["text"]
        
    models_change_flag=False # 模型调用一次后,标志位复位,标记模型的新状态; 否则,后续每次都会重复重新调用模型;
            
    return contents, models_change_flag

def audio_source(source, url):
    if source == "upload":
        inp_url = gr.Textbox(visible=False)
        out = gr.Audio(
            value=None,
            source="upload",
        )
    elif source == "microphone":
        inp_url = gr.Textbox(visible=False)
        out = gr.Audio(
            value=None,
            source="microphone",
        )
    elif source == "url":
        inp_url = gr.Textbox(visible=True)
        try:
            out = gr.Audio(
                value=url, 
                source=None)
        except:  # 异常时,恢复upload上传来源
            out = gr.Audio(value=None,source="upload")

        # inp_url.input(lambda x: x, inp_url,out)
    else:
        print("invalid audio source")
    return inp_url, out


def model_checkbox(models_change_flag):
    models_change_flag = True
    return models_change_flag


if __name__ == "__main__":

    model_selected = ["VAD", "PUNC"] # 模型选择缺省值;
    # inference_pipeline = Paraformer_longaudio_model() # 使用缺省值生成模型;
    
    with gr.Blocks(
        theme="soft",
        title="UniASR语音实时识别",
    ) as demo:
        gr.Markdown(
            """[**语音识别**](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell1-vocab8404-pytorch/summary)              
               [**长音频离线识别模型**](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
            > 1. 录音,或者上传音频,单声道,16K采样率音频会减少运行时间; 尽管如此,其它格式会自动转换成; 
            > 1. 选择是否使用 vad(voice activity detection), punc(标点), lm (NNLM) 诸模型
            > 1. 选择输出是否带时间戳;
            > 1. 点击,"一键识别",输出语音文字
            """
        )
        with gr.Row():
            with gr.Column(variant="panel"):
                inp0 = gr.Radio(
                    choices=["microphone", "upload", "url"],
                    value="upload",
                    type="value",
                    label="选择音频来源",
                    show_label=True,
                )
                inp_url = gr.Textbox(
                    lines=2,
                    placeholder="https://",
                    label="输入音频链接",
                    show_label=True,
                    show_copy_button=True,
                    value="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_speaker_demo.wav",
                    visible=False # 初始就让其不可现,仅当点击url时,才可见
                )                               
            
                inp1 = gr.Audio(
                    source='upload',
                    type="numpy",
                    show_label=True,
                    interactive=True,
                )
                inp0.change(audio_source, [inp0, inp_url], [inp_url,inp1],show_progress=True)
                inp_url.submit(audio_source,[inp0,inp_url],[inp_url,inp1],show_progress=True)
                
                with gr.Row(variant="panel"):
                    inp2 = gr.CheckboxGroup(
                        ["VAD", "PUNC", "NNLM"],
                        label="开启以下功能:",
                        value=model_selected,
                        show_label=True,
                    )

                    models_change_flag_var = gr.State(False) # 缺省模型没有被选择;
                    # use_timestamp_var = gr.State(True) # 缺省use_timestamp=True
                    inp3 = gr.Checkbox(value=True, label="时间戳", show_label=True)
                    
                    inp2.select(model_checkbox, models_change_flag_var, models_change_flag_var)
                    # inp3.select(use_timestamp_checkbox, use_timestamp_var, use_timestamp_var)
                    
            with gr.Column(variant="panel"):
                out0 = gr.Textbox(
                    lines=6,
                    placeholder="语音识别为:",
                    label="识别文本......",
                    show_label=True,
                    show_copy_button=True,
                )

        with gr.Row(variant="panel"):
            submit = gr.Button(value="一键识别", variant="primary")
            submit.click(RUN, [ inp1, inp2, models_change_flag_var, inp3], [out0,models_change_flag_var])
            clear = gr.Button(value="清除", variant="primary")

            clear.click(lambda: "", outputs=out0)

        demo.queue()
        demo.launch(show_error=True, share=True, debug=True)
