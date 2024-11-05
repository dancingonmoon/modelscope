import gradio as gr

def echo(message, history):
    return message["text"]


if __name__ == "__main__":

    demo = gr.ChatInterface(
        fn=echo,
        type="messages",
        examples=[{"text": "hello"}, {"text": "hola"}, {"text": "merhaba"}],
        title="Echo Bot",
        multimodal=True,
    )
    demo.launch()