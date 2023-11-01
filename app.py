# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
import gradio as gr
import numpy as np

# import logging

import ParaformerOffline
import ParaformerOnline

model_choices = ["VAD", "PUNC", "NNLM", "HotWords", "TimeStamp"]  # 模型选项;
model_selected = ["VAD", "PUNC"]  # 模型缺省配置


if __name__ == "__main__":
    with gr.Blocks(
        theme="soft",
        title="Paraformer模型,语音实时/离线识别",
    ) as demo:
        # 声明 offline 变量:
        # models_change_flag_var = gr.State(False)  # 缺省模型没有被选择;
        # 声明 online 变量:
        # speech_txt_var = gr.State("")
        # vad_stream_var = gr.State(np.zeros(shape=(1,), dtype=np.float32))

        gr.Markdown(
            """
            [**语音识别**](https://alibaba-damo-academy.github.io/FunASR/en/)              
            **Tab页,支持Offline与Online**
            > + ***Tab: Offline:***   
            >> 1. 录音,或者上传音频,单声道,16K采样率音频会减少运行时间; 尽管如此,其它格式会自动转换; 
            >> 1. 选择是否使用 vad(voice activity detection), punc(标点), lm(NNLM), HotWords, 以及TimeStamp, 重新加载模型;
            >> 1. 点击,"一键识别",输出语音文字
            >> 1. <font color='orange' face='lisu'>时间戳生成步骤:a)asr+tp;b)punc;c)拼接时间戳与句子</font>
            > + ***Tab: Online:***   
            >> 1. Stream Online模式,仅支持录音;
            >> 2. 选择是否在实时的同时,打开VAD; 由于VAD会压缩语音,剔除静音,会导致延时;
            >> 3. 录音结束时,会自动离线punctuation一次;
            """
        )
        with gr.Tab(label="Offline"):
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
                        ParaformerOffline.audio_source,
                        [inp0, inp_url],
                        [inp_url, inp1],
                        show_progress=True,
                        api_name="radio2audio_source",
                    )
                    inp_url.submit(
                        ParaformerOffline.audio_source,
                        [inp0, inp_url],
                        [inp_url, inp1],
                        show_progress=True,
                        api_name="url2audio_source",
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
                            ParaformerOffline.models_checkbox_on_select,
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
                    ParaformerOffline.RUN,
                    [inp1, inp2, models_change_flag_var, inp5],
                    [out0, models_change_flag_var],
                    api_name="offline_RUN",
                )
                clear = gr.Button(value="清除", variant="primary")

                clear.click(lambda: "", outputs=out0, api_name="offline_lambda")

        with gr.Tab(label="Online"):
            with gr.Row(variant="panel"):
                with gr.Column():
                    audio_stream = gr.Audio(
                        source="microphone",
                        type="filepath",
                        label="请录音并实时说话",
                        streaming=True,
                        show_label=True,
                        interactive=True,
                    )
                    vad_flag = gr.Checkbox(
                        value=False,
                        label="是否识别之前打开VAD. Note: 会有较大延时",
                        show_label=True,
                    )
                with gr.Column(variant="panel"):
                    speech_txt_var = gr.State("")
                    out = gr.Textbox(
                        lines=2,
                        placeholder="实时识别....",
                        label="实时识别文本输出:",
                        show_label=True,
                        show_copy_button=True,
                    )
                    clear = gr.Button(value="清除", variant="primary")
                    clear.click(
                        lambda: ("", ""),
                        outputs=[out, speech_txt_var],
                        api_name="online_lambda",
                    )  # 存放speech_txt的变量也清零

                    vad_stream_var = gr.State(np.zeros(shape=(1,), dtype=np.float32))
                    audio_stream.stream(
                        ParaformerOnline.RUN,
                        [audio_stream, speech_txt_var, vad_stream_var, vad_flag],
                        [out, speech_txt_var, vad_stream_var],
                        api_name="online_RUN",
                    )
                    audio_stream.stop_recording(
                        ParaformerOnline.punc_offline,
                        inputs=speech_txt_var,
                        outputs=out,
                    )  # 停止录音,离线打标点;
                    audio_stream.start_recording(
                        ParaformerOnline.continue_recording,
                        inputs=out,
                        outputs=speech_txt_var,
                    )  #

    demo.queue()
    demo.launch(show_error=True, share=True, debug=True)
