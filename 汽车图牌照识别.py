from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
# import cv2 #cv2在headless云端无法显示图片,弃用
import matplotlib.pyplot as plt
import math

import gradio as gr
import numpy as np


# photo_path = '汽车带牌照1.jpg'

def VehiclePlate_Recognition(photo_path):
    """
    输入photo_data,可以为文件路径; 或者numpy; gradio输入图片后,会自动输出numpy
    """

    # 给定一张图片，检测出图中车牌的位置并输出车的类型（比如小汽车，挂车，新能源车等）
    license_plate_detection = pipeline(Tasks.license_plate_detection, model='damo/cv_resnet18_license-plate-detection_damo') 
    plate = license_plate_detection(photo_path)
    # print(plate)
    
    if isinstance(photo_path, str):
        # 读图,获取图片array:
        img = plt.imread(photo_path)
        # 云端显示图片
        # plt.imshow(img)
    elif isinstance(photo_path, np.ndarray):
        img = photo_path # gradio输出的image的numpy结构为: (height, width, 3)
        
    # 获取牌照的最大面积截图矩阵:
    widths = plate['polygons'][0,::2] # 四坐标x轴(width)
    heights = plate['polygons'][0,1::2] # 四坐标y轴(heights)
    print(widths,heights)
    H_up = math.floor(min(heights))
    H_down = math.ceil(max(heights))
    W_left = math.floor(min(widths))
    W_right = math.ceil(max(widths))
    
    plate_crop = img[H_up:H_down, W_left:W_right]
    # print(H_up,H_down,W_left,W_right)
    # 云端显示牌照截图:
    # plt.imshow(plate_crop)
    
    # 给定一张文本图片，识别出图中所含文字并输出对应字符串
    ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-licenseplate_damo')
    
    recognized_plate = ocr_recognition(plate_crop)
    return [plate['text'], recognized_plate['text']]

def image_source(img_src):
    if img_src == 'upload':
        return gr.update(source='upload',label="请上传图片")
    if img_src == 'webcam':
        return gr.update(source='webcam',label="请拍照")
    

if __name__ == "__main__":
  
    with gr.Blocks(
        theme="soft",
        title="传入汽车图,获取汽车牌照,以及汽车类别",
    ) as demo:
        gr.Markdown(
            """[车牌照识别](https://modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-licenseplate_damo/summary)
            > 1. 拍照,或者上传带有汽车牌照的汽车图片
            > 1. 点击,"提交",输出汽车牌照,汽车类别            
            """)
        inp0 = gr.Radio(
                    choices=["camera", "upload"],
                    value="upload",
                    type="value",
                    label="选择图片来源",
                    show_label=True,
                )
        inp1 = gr.Image(
                    source="upload",
                    type="numpy",
                    show_label=True,
                    interactive=True,
                )
        img_numpy = inp0.change(image_source, inputs=inp0, outputs=inp1)
        out = gr.Text(label='车辆类别,识别车牌号', show_copy_button=True,show_label=True,
                     placeholder="['小型汽车'],['浙A88888']")
        submit_button = gr.Button(value='提交')
        submit_button.click(VehiclePlate_Recognition,img_numpy,out)
        
        
        demo.queue()
        demo.launch(show_error=True, share=True)