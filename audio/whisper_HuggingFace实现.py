import transformers
import gradio as gr

transformers.BarkModel.

# audio_path = r"H:\music\Music\2.mp3"
model = whisper.load_model("small")

# result = model.transcribe(audio_path,verbose=True,word_timestamps=True,
#                           initial_prompt='以下为普通话.描述的是一段京剧唱白,请给每句添加标点符号',
#                           language='Chinese',fp16=False,)
# print(result["text"])


def whisper_fn(
    audio_path,
    initial_prom,
    translate_flag=False,
    verbose=False,
    # word_timestamps=False,
    language="Chinese",
    GPU=False,
):
    task = "translate" if translate_flag else None
    if language == "unknown":
        language = None
    elif task == "translate":  # 当选择翻译成英文时,language这个参数必须设置成None,否则会出现几种语言胡乱翻译
        language = None
    else:
        pass

    result = model.transcribe(
        audio_path,
        task=task,
        verbose=verbose,
        # word_timestamps=word_timestamps,
        initial_prompt=initial_prom,
        language=language,
        fp16=GPU,
    )
    out1 = result["text"]
    # out1 = result
    segments = result["segments"]
    time_texts = ""
    for segment in segments:
        time_texts += f"{segment['start']}->{segment['end']}:  {segment['text']}\n"

    out2 = time_texts
    out3 = f"language:  {result['language']};"
    out4 = result  # 打印全部输出
    return out1, out2, out3, out4


def audio_source(source="microphone"):
    if source == "microphone":
        inp = gr.update(source="microphone", label="请录音:", visible=True)
    elif source == "upload":
        inp = gr.update(source="upload", label="请上传音频:", visible=True)
    elif source == "None":
        inp = gr.update(visible=False)
    else:
        raise ValueError("Invalid audio source")
    return inp


if __name__ == "__main__":
    # model = whisper.load_model("small")

    with gr.Blocks(
        theme="soft",
        title="语音识别以及英文翻译",
    ) as demo:
        gr.Markdown("## 语音识别+翻译英文")
        with gr.Row():
            with gr.Column():
                inp_ = gr.Radio(
                    choices=["microphone", "upload", "None"],
                    value="microphone",
                    type="value",
                    label="选择音频源",
                    show_label=True,
                )
                inp1 = gr.Audio(
                    source="microphone",
                    type="filepath",
                    label="请录音:",
                    show_label=True,
                    interactive=True,
                )
                inp_.change(audio_source, inputs=inp_, outputs=inp1)
                inp2 = gr.Textbox(
                    lines=2,
                    label="initial prompt",
                    show_label=True,
                    show_copy_button=True,
                    placeholder="提示语, 例如:以下为普通话.描述的是一段京剧唱白,请给每句添加标点符号",
                )
                inp6 = gr.Dropdown(
                    value="Chinese",
                    choices=["unknown", "chinese", "english", "russian", "persian"],
                    type="value",
                    label="Language",
                    show_label=True,
                )
                inp3 = gr.Checkbox(value=False, label="翻译成英文", show_label=True)
                inp4 = gr.Checkbox(value=False, label="verbose", show_label=True)
                inp5 = gr.Checkbox(value=False, label="打印全部输出")

                inp7 = gr.Checkbox(value=False, label="CPU/GPU", show_label=True)
            with gr.Column():
                out1 = gr.Text(
                    lines=2,
                    placeholder="音频内容为:",
                    show_copy_button=True,
                    label="text",
                    show_label=True,
                )
                out2 = gr.Text(
                    lines=5, show_copy_button=True, label="带时间戳的文本", show_label=True
                )
                out3 = gr.Text(label="识别语言", show_copy_button=True)
                out4 = gr.Text(
                    lines=5, label="打印全部输出", show_copy_button=True, visible=False
                )
                inp5.change(
                    lambda x: gr.update(visible=True)
                    if x
                    else gr.update(visible=False),
                    inputs=inp5,
                    outputs=out4,
                )

        button = gr.Button("RUN")
        button.click(
            whisper_fn,
            inputs=[
                inp1,
                inp2,
                inp3,
                inp4,
                # inp5,
                inp6,
                inp7,
            ],
            outputs=[out1, out2, out3, out4],
        )

    demo.queue()
    demo.launch(show_error=True, share=True)
