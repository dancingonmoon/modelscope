from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import gradio as gr
import numpy as np
# import soundfile

def Paraformer_longaudio_model(use_vad_model=True,use_punc_model=True,use_lm_model=False):
    
    if use_vad_model:
        vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch'
    else:
        vad_model=''
        
    if use_punc_model:
        punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
    else:
        punc_model=''
        
    if use_lm_model:
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            # defaults to combine VAD, ASR and PUNC
            model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            vad_model=vad_model,
            punc_model=punc_model,
            lm_model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch',
            lm_weight=0.15,
            beam_size=10,
            model_revision=None,
            )
    else:
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            # defaults to combine VAD, ASR and PUNC
            model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            vad_model=vad_model,
            punc_model=punc_model,
            model_revision=None,
            )           
        
    return inference_pipeline #这里先输出模型, 以避免后续模型重复生成;

def Paraformer_longaudio_recognition(inference_pipeline, audio_data,use_timestamp=True,):
    """
    audio_data: 为输入音频,为gr.Audio输出,为元组: (int sample rate, numpy.array for the data),二进制数据(bytes);url    
    """
    param_dict = {'use_timestamp': use_timestamp}
    # param_dict['use_timestamp'] = use_timestamp
    sample_rate, waveform = audio_data
    rec_result = inference_pipeline(audio_in=waveform)#, param_dict=param_dict)
    return rec_result

def audio_source(source):
    if source == 'upload':
        out = gr.Audio(source="upload",)
    elif source == 'microphone':
        out = gr.Audio(source='microphone',)
    else:
        print('invalid source')
    return out

def model_checkbox(model_selected):
    use_vad_model = True if "VAD" in model_selected else False
    use_punc_model = True if "PUNC" in model_selected else False
    use_lm_model = True if "NNLM" in model_selected else False
    inference_pipeline = Paraformer_longaudio_model(use_vad_model=use_vad_model,
                                                    use_punc_model=use_punc_model,
                                                    use_lm_model=use_lm_model)
    return inference_pipeline
    
def run(audio_data,model_selected, use_timestamp=True):
    """
    audio_data: 为输入音频,为gr.Audio输出,为元组: (int sample rate, numpy.array for the data),二进制数据(bytes);url    
    """
    sample_rate, waveform = audio_data
    waveform = waveform.astype(np.float32) # gr.Audio转换的为(sample rate in Hz, audio data as a 16-bit int array);而输入pipeline需要float32, 双声道转换成单声道;
    if len(waveform.shape)>=1: # 说明是双声道
        waveform = np.mean(waveform,axis=1) # 转换成单声道
    # waveform, sample_rate = soundfile.read(audio_data)
    inference_pipeline = model_checkbox(model_selected)
    result = inference_pipeline(waveform, use_timestamp=use_timestamp) #, param_dict)   
    
    # 读出内容:
    contents = ''
    for dic in result['sentences']:
        contents +=  f"{dic['start']}->{dic['end']} : {dic['text']} \n"
    
    
    return contents

if __name__ == "__main__":
  
    # inference_pipeline = Paraformer_longaudio_model()
    with gr.Blocks(
        theme="soft",
        title="UniASR语音实时识别",
    ) as demo:
        gr.Markdown(
            """[**语音识别**](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell1-vocab8404-pytorch/summary)
                [**长语音模型**](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
            > 1. 录音,或者上传音频wav格式
            > 1. 选择是否使用 vad(voice activity detection), punc(标点), lm (NNLM) 诸模型
            > 1. 点击,"一键识别",输出语音文字
            """)
        with gr.Row():
            with gr.Column(variant='panel'):
                inp0 = gr.Radio(
                    choices=["microphone", "upload"],
                    value="upload",
                    type="value",
                    label="选择音频来源",
                    show_label=True,
                    )
                inp1 = gr.Audio(
                    source="upload",
                    type="numpy",
                    show_label=True,
                    interactive=True,
                    )
                inp0.change(audio_source,inp0, inp1)
                with gr.Row(variant='panel'):
                    inp2 = gr.CheckboxGroup(["VAD", "PUNC","NNLM"], label="请选择是否在ASR模型外,包含下面模型",
                                        value=["VAD", "PUNC"],show_label=True )
                    
                    # model_state = gr.State() # 缺省模型送入临时State, 因为模型不是可以deepcopy的对象,所以gr.State不支持暂存模型
                    # inference_pipeline = inp2.select(model_checkbox, inp2,inp2)
                     
                    inp3 = gr.Checkbox(value=True,label='时间戳', show_label=True)                
            with gr.Column(variant='panel'):
                out0 = gr.Textbox(lines=6,placeholder='语音识别为:', label='识别结果:',show_label=True,show_copy_button=True)
            
        button = gr.Button(value='一键识别', variant='primary')
        button.click(run, [inp1,inp2,inp3], out0)
        # button.click(debug,model_state, out0)
        
            
        demo.queue()
        demo.launch(show_error=True, share=True, debug=True)