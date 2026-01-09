import os
from typing import Literal
import base64
import requests
import json
from pathlib import Path
import langextract as lx
from langextract.providers.openai import OpenAILanguageModel
from langextract.prompt_validation import PromptValidationLevel

# pdf_path = Path(r"E:/Working Documents/Eastcom/市场资料/彭博行业研究_全球关税展望.pdf")
pdf_path = Path(r"E:/Working Documents/Eastcom/Russia/Igor/专网/LeoTelecom/yaml/Delivery/收款/PI_251110.pdf")

# 创建输出目录
OUTPUT_DIR = Path("./data/lx_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# PaddleOCR_VL 读取并转化PDF文件
API_URL = 'https://n9pfq0l2acc5zeie.aistudio-app.com/layout-parsing'  # PaddleOCR_VL
TOKEN = os.getenv('paddleOCR_TOKEN')


def paddleOCR_read(paddle_api_url: str = 'https://n9pfq0l2acc5zeie.aistudio-app.com/layout-parsing',
                   paddle_token: str = None,
                   input_path: str | Path = None,
                   input_file_type: Literal['pdf', 'image'] = 'pdf',
                   output_dir: str | Path = None,
                   useDocOrientationClassify: bool = False,  # 图像方向纠正;
                   useDocUnwarping: bool = False,  # 图片扭曲纠正;
                   useChartRecognition: bool = True,  # 图表识别;
                   useLayoutDetection: bool = False,  # 版本分析
                   promptLabel: str = None,
                   output_format: Literal['json', 'markdown'] = 'json',
                   ):
    """
    使用PaddleOCR在线API服务，将PDF或者image阅读并转化成json或者markdown文件
    API约束：
        0. 参考API文档: https://ai.baidu.com/ai-doc/AISTUDIO/2mh4okm66
        1. 每日对同一模型的解析上限为3000页，超出将返回429错误。如有更高调用需求，可以通过问卷免费申请白名单。
        2. 单个文件大小不限制，但为避免处理超时，建议每个文件不超过100页。若超过100页，API只解析前100页，后续页将被忽略。
    :param paddle_api_url:
    :param paddle_token:
    :param input_path: 服务器可访问的图像文件或PDF文件的URL，或上述类型文件内容的Base64编码结果。默认对于超过10页的PDF文件，只有前10页的内容会被处理。
    :parm input_file_type: 文件类型。0表示PDF文件，1表示图像文件。若请求体无此属性，则将根据URL推断文件类型。
    :param output_dir:
    :param useDocOrientationClassify: 是否在推理时使用文本图像方向矫正模块，开启后，可以自动识别并矫正 0°、90°、180°、270°的图片。
    :param useDocUnwarping: 是否在推理时使用文本图像矫正模块，开启后，可以自动矫正扭曲图片，例如褶皱、倾斜等情况。
    :param useChartRecognition: 是否在推理时使用图表解析模块，开启后，可以自动解析文档中的图表（如柱状图、饼图等）并转换为表格形式，方便查看和编辑数据。
    :param useLayoutDetection: 是否在推理时使用版面区域检测排序模块，开启后，可以自动检测文档中不同区域并排序。
    :param promptLabel: VL模型的 prompt 类型设置，当且仅当 useLayoutDetection=False 时生效。
    :param output_format: 选择以json或者markdown格式存储
    :param output_dir: 本地存储的文件夹
    :return: 返回阅读出的json对象
    """
    if input_path:
        input_path = Path(input_path)
    if output_dir:
        output_dir = Path(output_dir)
    if paddle_token is None:
        paddle_token = os.getenv('paddleOCR_TOKEN')
    with open(input_path, "rb") as pdf_file:
        pdf_file_bytes = pdf_file.read()
        pdf_file_data = base64.b64encode(pdf_file_bytes).decode("ascii")

    headers = {
        "Authorization": f"token {paddle_token}",
        "Content-Type": "application/json"
    }

    if input_file_type == 'pdf':
        fileType = 0
    elif input_file_type == 'image':
        fileType = 1
    else:
        fileType = None

    required_payload = {
        "file": pdf_file_data,
        "fileType": fileType  # PDF,set 0; image, set 1
    }

    optional_payload = {
        "useDocOrientationClassify": useDocOrientationClassify,  # 图像方向纠正;
        "useDocUnwarping": useDocUnwarping,  # 图片扭曲纠正;
        "useChartRecognition": useChartRecognition,  # 图表识别;
        "useLayoutDetection": useLayoutDetection,  # 版本分析
    }
    if useLayoutDetection is False:
        optional_payload["promptLabel"] = promptLabel

    payload = {**required_payload, **optional_payload}
    response = requests.post(paddle_api_url, json=payload, headers=headers)
    # print(f"status_code: {response.status_code}")
    assert response.status_code == 200
    result = response.json()["result"]

    if output_format == 'json':
        if output_dir:  # 保存JSON结果到本地文件
            json_output_path = Path(output_dir, f"{input_path.stem}.json")
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"json document saved at {json_output_path}")
    elif output_format == 'markdown':
        for i, res in enumerate(result["layoutParsingResults"]):
            md_filename = Path(output_dir, f"{input_path.stem}_doc{i}.md")
            with open(md_filename, "w", encoding="utf-8") as md_file:
                md_file.write(res["markdown"]["text"])
            print(f"Markdown document saved at {md_filename}")
            for img_path, img in res["markdown"]["images"].items():
                full_img_path = Path(output_dir, img_path)
                os.makedirs(os.path.dirname(full_img_path), exist_ok=True)
                img_bytes = requests.get(img).content
                with open(full_img_path, "wb") as img_file:
                    img_file.write(img_bytes)
                print(f"Image saved to: {full_img_path}")
            for img_name, img in res["outputImages"].items():
                img_response = requests.get(img)
                if img_response.status_code == 200:
                    # Save image to local
                    filename = Path(output_dir, f"{img_name}_{i}.jpg")
                    with open(filename, "wb") as f:
                        f.write(img_response.content)
                    print(f"Image saved to: {filename}")
                else:
                    print(f"Failed to download image, status code: {img_response.status_code}")

    return result


result = paddleOCR_read(input_path=pdf_path,
                        input_file_type='pdf',
                        output_format='markdown', output_dir=OUTPUT_DIR)
