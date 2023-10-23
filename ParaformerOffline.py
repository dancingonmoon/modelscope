from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import gradio as gr
import numpy as np

import librosa
import time
import re

# large asr+vad+punc
# asr_model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
# large asr 仅仅
# asr_model = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
# large asr+vad+punc+spk分角色
# asr_model="damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn",  # 分角色语音识别
# large 热词 asr 仅仅:
asr_model = "damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
vad_model = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
punc_model = "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
# punc_model = "damo/punc_ct-transformer_cn-en-common-vocab471067-large"
lm_model = "damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch"
timestamp_model = "damo/speech_timestamp_prediction-v1-16k-offline"

output_dir = "./results"

model_choices = ["VAD", "PUNC", "NNLM", "HotWords", "TimeStamp"]  # 模型选项;
model_selected = ["VAD", "PUNC"]  # 模型缺省配置

inference_pipeline_punc = pipeline(
    task=Tasks.punctuation,
    model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    output_dir=output_dir,
)


def ms2strftime(timestamp):
    """
    将毫秒值转换为可读的时间格式
    timestamp: 为毫秒值
    """
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(timestamp / 1000))
    return formatted_time


def timeframe2idx_txt(txt, timestamp):
    """
    txt:为已经punc后,带标点与断句的文本将txt;
    timestamp: 为txt中每个字的时间戳的元组(起,止);
    输出表达式: 句子时间戳的起->止:句子
    """
    # 获取txt的句子列表:
    # pattern = r'[^。，；？！]+'
    pattern = r"[\w\s]+"
    find_list = re.findall(pattern, txt)
    # 获得移除标点符号后的txt:
    txt_NoPunc = re.sub(r"[^\w\s]+", "", txt)
    # 获得句子index列表:
    time_frame_list = []
    start = 0
    for i in find_list:
        # 每句移除标点符号
        i = re.sub(r"[^\w\s]+", "", i)
        start = txt_NoPunc.find(i, start)  # 因为可能有重复,每次检索从新位置开始
        end = start + len(i) - 1
        time_frame_list.append(start)
        start = end + 1
    time_frame = [timestamp[i] for i in time_frame_list]
    time_frame_list.append(-1)
    # 将句子时间戳列表最后一个字符的起始更换成句子的最终时间戳;
    last_end = timestamp[-1][1]
    time_frame[-1][0] = last_end
    # 输出时间戳-句子,表达式:
    out = []
    # 获得时间戳的最大长度,以显示时对齐:
    max_timestamp_len = len(str(timestamp[-1][1])) * 2 + 2

    for i in range(len(time_frame) - 1):
        time_frame_index = f"{time_frame[i][0]}->{time_frame[i+1][0]}"
        time_frame_txt = f"{txt_NoPunc[time_frame_list[i]:time_frame_list[i+1]]}"
        line_out = f"{time_frame_index:>{max_timestamp_len}}:  {time_frame_txt}"
        out.append(line_out)
        # print(line_out)

    return out


def Paraformer_longaudio_model(
    use_vad_model=True,
    use_punc_model=True,
    use_lm_model=False,
    use_hotword=False,
    hotword_txt="",
    use_timestamp=False,
):
    global asr_model
    global vad_model
    global punc_model
    global lm_model
    global timestamp_model

    vad_model = "" if not use_vad_model else vad_model
    punc_model = "" if not use_punc_model else punc_model

    config = dict(
        task=Tasks.auto_speech_recognition,
        model=asr_model,
        vad_model=vad_model,
        punc_model=punc_model,
        lm_model=lm_model,
        lm_weight=0.15,
        beam_size=10,
        # timestamp_model=timestamp_model,
        model_revision=None,
        output_dir=output_dir,
        # param_dict=param_dict
    )

    if not use_lm_model:
        config.pop("lm_model", None)
        config.pop("lm_weight", None)
        config.pop("beam_size", None)

    if use_hotword:
        # 添加自定义hotword
        if isinstance(hotword_txt, str):
            hotword_txt = re.sub(
                r"[\s\n,]+", "\n", hotword_txt
            )  # 将单个或者连续的空格/换行/逗号,替换成换行符
        else:
            hotword_txt = ""

        # param_dict['hotword'] = hotword_txt
        param_dict = dict(hotword=hotword_txt)
        config["param_dict"] = param_dict

    # 由于timestamp_models在级联方式下,处理比较复杂,对音频的长度有要求, 这里凡是use_timestamp=True,就只合并asr模型与timestamp_model:
    if use_timestamp:  # asr+tp,输出为无标点的文本,以及每个字的时间戳.
        config.pop("lm_model", None)
        config.pop("lm_weight", None)
        config.pop("beam_size", None)
        config.pop("vad_model", None)
        config.pop("punc_model", None)
        config["timestamp_model"] = timestamp_model

    inference_pipeline = pipeline(**config)

    return inference_pipeline  # 这里先输出模型, 以避免后续模型重复生成;


inference_pipeline = Paraformer_longaudio_model()  # 提前启动缺省模型.


def RUN(audio_data, model_selected, models_change_flag=False, hotword_txt=""):
    """
    audio_data: 为输入音频,为gr.Audio输出,为元组: (int sample rate, numpy.array for the data),二进制数据(bytes);url
    """
    print(f"hotword_txt={hotword_txt}")
    global inference_pipeline
    # 判断模型选择是否发生变化,并重新加载模型:

    use_timestamp = False  # 初始值

    if models_change_flag:
        use_vad_model = True if "VAD" in model_selected else False
        use_punc_model = True if "PUNC" in model_selected else False
        use_lm_model = True if "NNLM" in model_selected else False
        use_hotword = True if "HotWords" in model_selected else False
        use_timestamp = True if "TimeStamp" in model_selected else False

        inference_pipeline = Paraformer_longaudio_model(
            use_vad_model=use_vad_model,
            use_punc_model=use_punc_model,
            use_lm_model=use_lm_model,
            use_hotword=use_hotword,
            hotword_txt=hotword_txt,
            use_timestamp=use_timestamp,
        )

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

    result = inference_pipeline(waveform)

    # 读出内容:
    # contents = ""
    # if use_timestamp and "sentences" in result:  # result中得有"sentences"键
    #     for dic in result["sentences"]:
    #         start = ms2strftime(dic["start"])
    #         end = ms2strftime(dic["end"])
    #         contents += f"{start}->{end} : {dic['text']} \n"
    # else:
    #     contents = result["text"]

    # 识别结果文本处理:
    # 如果无时间戳,仅result['text'];如时间戳,则先1.punc;2.断句;3.匹配句与时间戳;4.输出.
    result_punc_tp = "起始为空"
    if "text" in result:
        if use_timestamp and "timestamp" in result:
            result_punc_txt = inference_pipeline_punc(result["text"])
            if len(result_punc_txt) != 0:
                result_punc_tp = timeframe2idx_txt(
                    result_punc_txt["text"], result["timestamp"]
                )
            print_output = "\n".join(result_punc_tp)
            result_output = print_output
        else:
            result_output = result["text"]
    else:  # 识别为空;(无识别结果)
        result_output = result

    models_change_flag = False  # 模型调用一次后,标志位复位,标记模型的新状态; 否则,后续每次都会重复重新调用模型;

    return result_output, models_change_flag


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
            out = gr.Audio(value=url, source=None)
        except:  # 异常时,恢复upload上传来源
            out = gr.Audio(value=None, source="upload")

        # inp_url.input(lambda x: x, inp_url,out)
    else:
        inp_url = gr.Textbox()
        out = gr.Audio()
        print("invalid audio source")

    return inp_url, out


def models_checkbox_on_select(env: gr.SelectData):
    model_ticked = env.value
    model_ticked_bool = env.selected

    if model_ticked == "HotWords":
        hotword_txt = gr.Textbox(visible=model_ticked_bool)
    else:
        hotword_txt = gr.Textbox()  # 保持不变

    models_change_flag_var = True

    return models_change_flag_var, hotword_txt


if __name__ == "__main__":
    with gr.Blocks(
        theme="soft",
        title="UniASR语音实时识别",
    ) as demo:
        gr.Markdown(
            """[**语音识别**](https://alibaba-damo-academy.github.io/FunASR/en/)              
               [**长音频离线识别模型**](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
            > 1. 录音,或者上传音频,单声道,16K采样率音频会减少运行时间; 尽管如此,其它格式会自动转换; 
            > 1. 选择是否使用 vad(voice activity detection), punc(标点), lm(NNLM), HotWords, 以及TimeStamp, 重新加载模型;
            > 1. 点击,"一键识别",输出语音文字
            > 1. <font color='orange' face='lisu'>时间戳生成步骤:a)asr+tp;b)punc;c)拼接时间戳与句子<font>
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
                    visible=False,  # 初始就让其不可现,仅当点击url时,才可见
                )

                inp1 = gr.Audio(
                    source="upload",
                    type="numpy",
                    show_label=True,
                    interactive=True,
                )
                inp0.change(
                    audio_source, [inp0, inp_url], [inp_url, inp1], show_progress=True
                )
                inp_url.submit(
                    audio_source, [inp0, inp_url], [inp_url, inp1], show_progress=True
                )

                with gr.Row(variant="panel"):
                    inp2 = gr.CheckboxGroup(
                        model_choices,
                        label="开启以下功能:",
                        value=model_selected,
                        show_label=True,
                    )

                    models_change_flag_var = gr.State(False)  # 缺省模型没有被选择;
                    # use_timestamp_var = gr.State(True)  # 缺省use_timestamp=True
                    # inp3 = gr.Checkbox(value=True, label="时间戳", show_label=True)
                    # inp4 = gr.Checkbox(value=False, label="添加热词", show_label=True)

                with gr.Row(variant="panel"):
                    inp5 = gr.Textbox(
                        lines=1,
                        placeholder="请输入热词,以空格,或者分号间隔,每个热词少于10个字:",
                        label="热词表",
                        show_label=True,
                        interactive=True,
                        visible=False,
                    )
                    # inp4.select(use_hotword_checkbox, inp4, inp5)  # 控制visible
                    inp2.select(
                        models_checkbox_on_select,
                        None,
                        [models_change_flag_var, inp5],
                    )

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
            submit.click(
                RUN,
                [inp1, inp2, models_change_flag_var, inp5],
                [out0, models_change_flag_var],
            )
            clear = gr.Button(value="清除", variant="primary")

            clear.click(lambda: "", outputs=out0)

        demo.queue()
        demo.launch(show_error=True, share=True, debug=True)
