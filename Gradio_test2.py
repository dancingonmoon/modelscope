import gradio as gr

# webcam_enabled = False
# webcam_mirrored = False


# def webcam_toggle():
#     global webcam_enabled
#     webcam_enabled = not webcam_enabled
#     # source = 'webcam' if webcam_mirrored else 'upload'
#     out = gr.update(source='webcam')
# out = {
#     "value": None,
#     "source": "webcam",
#     "__type__": "update",
# }
# if webcam_enabled:
#     return gr.Image(source='webcam')
# else:
#     return gr.Image(source='upload')
# return out


# return {
#     "value": None,
#     "source": "webcam" if webcam_enabled else "upload",
#     "__type__": "update",
# }


# def webcam_mirror_toggle():
#     global webcam_mirrored
#     webcam_mirrored = not webcam_mirrored
#     return {"mirror_webcam": webcam_mirrored, "__type__": "update"}


# with gr.Blocks(theme="soft") as demo:
#     with gr.Row():
#         webcam_enabled_checkbox = gr.Checkbox(
#             label="webcam_enabled_toggle", show_label=True
#         )
#         # webcam_mirrored_checkbox = gr.Checkbox(
#         #     label="webcam_mirrored_toggle", show_label=True
#         # )
#     with gr.Row():
#         input_image = gr.Image(
#             source="upload", mirror_webcam=False, type="numpy", tool="sketch", label='输入照片'
#         )
#
#     webcam_enabled_checkbox.select(fn=webcam_toggle, inputs=None, outputs=input_image)
# webcam_mirrored_checkbox.select(
#     fn=webcam_mirror_toggle, inputs=None, outputs=input_image
# )


# import gradio as gr

with gr.Blocks() as demo:
    with gr.Tab("File"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Input Image")
                run = gr.Button("Run")
            with gr.Column():
                output = gr.Image(label="Input Image")
    with gr.Tab("File"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Input Image", source="webcam")
                run = gr.Button("Run")
            with gr.Column():
                output = gr.Image(label="Input Image")

demo.launch()

if __name__ == "__main__":
    demo.queue().launch(
        debug=True,
    )
