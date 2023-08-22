import whisper
import gradio as gr

# audio_path = r"H:\music\Music\2.mp3"
model = whisper.load_model("small")


# result = model.transcribe(audio_path,verbose=True,word_timestamps=True,
#                           initial_prompt='以下为普通话.描述的是一段京剧唱白,请给每句添加标点符号',
#                           language='Chinese',fp16=False,)
# print(result["text"])

def test(audio_path, initial_prom, verbose=False, word_timestamps=False, language='Chinese', GPU=False):
    result = model.transcribe(audio_path, verbose=verbose, word_timestamps=word_timestamps,
                              initial_prompt=initial_prom,
                              language=language, fp16=GPU, )
    out1 = result['text']
    segments = result["segments"][0]
    out2 = segments['tokens']
    out3 = f"'temperature':  {segments['temperature']};\n'avg_logprob':  {segments['avg_logprob']:.4f};\n'compression_ratio':  {segments['compression_ratio']:.4f};\n'no_speech_prob':  {segments['no_speech_prob']:.4f};\n'language':  {result['language']};"
    return out1, out2, out3


if __name__ == '__main__':
    # model = whisper.load_model("small")
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                inp1 = gr.Audio(source='microphone', type='filepath', label='请录音:', show_label=True)
                inp2 = gr.Textbox(lines=2, label='initial prompt', show_label=True, show_copy_button=True,
                                  placeholder='提示语, 例如:以下为普通话.描述的是一段京剧唱白,请给每句添加标点符号')
                inp3 = gr.Checkbox(value=False, label='verbose', show_label=True)
                inp4 = gr.Checkbox(value=False, label='word_timestamps')
                inp5 = gr.Dropdown(value="Chinese", choices=['Chinese', 'English', 'Russian', 'Farsi'],
                                   label='Language', show_label=True)
                inp6 = gr.Checkbox(value=False, label='CPU/GPU', show_label=True)
            with gr.Column():
                out1 = gr.Text(lines=5,placeholder='音频内容为:', show_copy_button=True, label='text', show_label=True)
                out2 = gr.Text(show_copy_button=True, label='tokens', show_label=True)
                out3 = gr.Text(label='参数标识', show_copy_button=True)

        button = gr.Button('RUN')
        button.click(test, inputs=[inp1, inp2, inp3, inp4, inp5, inp6], outputs=[out1, out2, out3])

    demo.queue()
    demo.launch(show_error=True, share=True)
