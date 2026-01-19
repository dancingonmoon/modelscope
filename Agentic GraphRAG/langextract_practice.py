import os
from pathlib import Path
from typing import Literal
import textwrap
import base64
import requests
import json
from pathlib import Path
import langextract as lx
from langextract.providers.openai import OpenAILanguageModel
from langextract.prompt_validation import PromptValidationLevel
from langextract.core.data import Document

# pdf_path = Path(r"E:/Working Documents/Eastcom/市场资料/彭博行业研究_全球关税展望.pdf")
pdf_path = Path(r"E:/Working Documents/Eastcom/Russia/Igor/专网/LeoTelecom/yaml/Delivery/收款/PI_251110.pdf")

# 创建输出目录
OUTPUT_DIR = Path("./data/lx_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# PaddleOCR_VL 读取并转化PDF文件
API_URL = 'https://n9pfq0l2acc5zeie.aistudio-app.com/layout-parsing'  # PaddleOCR_VL
TOKEN = os.getenv('paddleOCR_TOKEN')


def paddleOCR_API(paddle_api_url: str = 'https://n9pfq0l2acc5zeie.aistudio-app.com/layout-parsing',
                  paddle_token: str = None,
                  input_path: str | Path = None,
                  input_file_type: Literal['pdf', 'image'] = 'pdf',
                  output_dir: str | Path = None,
                  useDocOrientationClassify: bool = False,  # 图像方向纠正;
                  useDocUnwarping: bool = False,  # 图片扭曲纠正;
                  useChartRecognition: bool = True,  # 图表识别;
                  useLayoutDetection: bool = False,  # 版本分析
                  visualize: bool = False,  # 支持返回可视化结果图及处理过程中的中间图像。
                  promptLabel: str = None,
                  output_format: Literal['json', 'markdown'] = 'json',
                  ):
    """
    使用PaddleOCR在线API服务，将PDF或者image阅读并转化成json或者markdown格式文件指定本地目录存储，或者返回json对象
    API约束：
        0. 参考API文档: https://ai.baidu.com/ai-doc/AISTUDIO/2mh4okm66
        1. 每日对同一模型的解析上限为3000页，超出将返回429错误。如有更高调用需求，可以通过问卷免费申请白名单。
        2. 单个文件大小不限制，但为避免处理超时，建议每个文件不超过100页。若超过100页，API只解析前100页，后续页将被忽略。
    :param paddle_api_url:
    :param paddle_token:
    :param input_path: 服务器可访问的图像文件或PDF文件的URL，或上述类型文件内容的Base64编码结果。默认对于超过10页的PDF文件，只有前10页的内容会被处理。
    :parm input_file_type: 文件类型。0表示PDF文件，1表示图像文件。若请求体无此属性，则将根据URL推断文件类型。
    :param useDocOrientationClassify: 是否在推理时使用文本图像方向矫正模块，开启后，可以自动识别并矫正 0°、90°、180°、270°的图片。
    :param useDocUnwarping: 是否在推理时使用文本图像矫正模块，开启后，可以自动矫正扭曲图片，例如褶皱、倾斜等情况。
    :param useChartRecognition: 是否在推理时使用图表解析模块，开启后，可以自动解析文档中的图表（如柱状图、饼图等）并转换为表格形式，方便查看和编辑数据。
    :param useLayoutDetection: 是否在推理时使用版面区域检测排序模块，开启后，可以自动检测文档中不同区域并排序。
    :param visualize: 支持返回可视化结果图及处理过程中的中间图像。开启此功能后，将增加结果返回时间。
    :param promptLabel: VL模型的 prompt 类型设置，当且仅当 useLayoutDetection=False 时生效。
    :param output_format: 选择以json或者markdown格式存储或return
    :param output_dir: 当None时，不以文件本地存储方式输出;否则，在指定目录，以指定的output_format格式文件存储输出;
    :return: 返回json对象，或者合并后的markdown文本;
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
        "visualize": visualize,
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
        return result
    elif output_format == 'markdown':
        markdown_text = ""
        for i, res in enumerate(result["layoutParsingResults"]):
            # 链接markdown文本，用于return
            markdown_text += res["markdown"]["text"]
            markdown_text += "\n"
            print(f"Markdown text for doc{i} completed. ")
            if output_dir:
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
                    print(f"Markdown Layout Image saved to: {full_img_path}")
                for img_name, img in res["outputImages"].items():
                    img_response = requests.get(img)
                    if img_response.status_code == 200:
                        # Save image to local
                        filename = Path(output_dir, f"{img_name}_{i}.jpg")
                        with open(filename, "wb") as f:
                            f.write(img_response.content)
                        print(f"Markdown Output Image saved to: {filename}")
                    else:
                        print(f"Failed to download image, status code: {img_response.status_code}")

        return markdown_text
    else:
        return result


# from langchain_deepseek import ChatDeepSeek
# deepseek_model = ChatDeepSeek(model='deepseek_chat')
#  langextract库,model需要使用改写过的OpenAILanguageModel,其中有不少自定义，例如多workers
deepseek_model = OpenAILanguageModel(
    # model_id='deepseek-chat',
    # api_key=os.getenv('DEEPSEEK_API_KEY'),
    # base_url="https://api.deepseek.com")
    model_id='qwen-flash',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

if __name__ == '__main__':
    # result = paddleOCR_API(input_path=pdf_path,
    #                        input_file_type='pdf',
    #                        output_format='markdown', output_dir=None)

    # input_text = "弗雷德里克•奥尼尔是19世纪涌入中国的新教传教士之一。这批人相信他们在中国能够取得成功。他们的对手、耶稣会会士一个世纪之前在中国传教的行动曾遭遇失败。他们要对抗的是疾病、孤单，还有难以接受新鲜事物的民众，没有几位传教士发现自己拯救了很多灵魂。罗伯特•马礼逊(Robert Morrison)是首批到中国的新教传教士之一。他曾有一句名言说，自己在27年的时间里仅仅让25人皈依新教。"
    pdf_path = Path(r"E:/Working Documents/Eastcom/市场资料/阿联酋投资问与答.pdf")


    # input_text = paddleOCR_API(input_path=pdf_path,
    #                            input_file_type='pdf',
    #                            output_dir=OUTPUT_DIR,
    #                            useChartRecognition=True,
    #                            useLayoutDetection=True,
    #                            output_format='markdown', )
    # ## 从存储在本地目录，已经OCR读出的MD文件中，按照Document Object的dataclass转换，并生成Iterable Document Object, 以送入lx.extract:
    # input_text = ""
    def load_md_Document_from_dir(pdf_path: str | Path, output_dir: str | Path):
        """
        从存储在本地目录，已经OCR读出的MD文件中，按照Document Object的dataclass转换，按doc的序列，生成Iterable Document dataclass, 以送入lx.extract:
        :param pdf_path:
        :param output_dir:
        :return: langextract.core.Document dataclass
        """
        for i in range(65):
            if not isinstance(pdf_path, Path):
                pdf_path = Path(pdf_path)
            md_path = Path(output_dir, f"{pdf_path.stem}_doc{i}.md")
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()
            md_Document = Document(text=md_text, document_id=f"doc{i}")
            yield md_Document


    input_text = load_md_Document_from_dir(pdf_path, OUTPUT_DIR)

    # 1 定义 LangExtract 的任务描述
    # langextract_prompt = """
    #                     从一个关于传教士历史的文本中提取以下信息：
    #                     - 时间
    #                     - 地点
    #                     - 组织
    #                     - 人物
    #                     - 事件
    #                     - 数量
    #                     - 背景
    #
    #                     要求：
    #                     1. 使用原文中的完整表述
    #                     2. 不要重复
    #                     3. 按出现顺序提取
    #                     """
    langextract_prompt = textwrap.dedent("""\
        从文档中提取以下结构化知识:

        - 实体: 人物、机构、地点、时间、概念、技术术语
        - 数据指标: 数值、百分比、统计数据
        - 关系描述: 实体之间的关系（合作、隶属、引用等）
        - 事件: 重要事件和行为

        要求:
        1. extraction_text 必须是原文的精确子串
        2. 为每个提取添加丰富的属性信息
        3. 关系类型必须在 attributes 中标注涉及的主体
        4. 保持原文出现顺序
        5. 输出严格遵循JSON格式，确保语法正确
        6. 每个JSON对象必须有冒号分隔符
        """)
    # 2 定义 LangExtract 的 Few-shot 示例
    # examples = [
    #     lx.data.ExampleData(
    #         text="1900年，排外的义和团运动令华北大乱。为了躲避义和团，奥尼尔逃到海参崴待了一年。",
    #         extractions=[
    #             lx.data.Extraction("时间", "1900年"),
    #             lx.data.Extraction("组织", "义和团"),
    #             lx.data.Extraction("地点", "华北"),
    #             lx.data.Extraction("人物", "奥尼尔"),
    #             lx.data.Extraction("事件", "奥尼尔逃到海参崴待了一年"),
    #             lx.data.Extraction("数量", "一年"),
    #             lx.data.Extraction("背景", "排外的义和团运动令华北大乱"),
    #         ]
    #     )
    # ]
    example_text = ("""阿联酋执行自由经济政策，无外汇管制，可自由汇进汇出，且跨境收支无币种限制，但须符合阿联酋政府的反洗钱规定。一般情况下，外商投资资本和利润回流不受限制。
    携带超过60,000迪拉姆现金或等值的其他货币、金融工具、贵金属或贵重宝石的乘客出境或进入阿联酋，必须向联邦身份、公民、海关和港口安全局（ICP）申报。""")
    example_extractions = [
        lx.data.Extraction(
            extraction_class="实体",
            extraction_text="阿联酋",
            attributes={"类型": "机构", "类别": "国家"}
        ),
        lx.data.Extraction(
            extraction_class="实体",
            extraction_text="迪拉姆现金",
            attributes={"类型": "货币", "类别": "现金", "单位": "迪拉姆", }
        ),
        lx.data.Extraction(
            extraction_class="数据指标",
            extraction_text="60,000迪拉姆",
            attributes={"类型": "货币金额", "单位": "迪拉姆", }
        ),
        lx.data.Extraction(
            extraction_class="实体",
            extraction_text="自由经济政策",
            attributes={"类型": "政策", "类别": "经济", "特征": "自由"}
        ),
        lx.data.Extraction(
            extraction_class="关系描述",
            extraction_text="阿联酋执行自由经济政策",
            attributes={"类型": "国家执行政策", "主体1": "阿联酋", "主体2": "自由经济政策", "关系": "政策执行"}
        ),
    ]
    examples = [
        lx.data.ExampleData(
            text=example_text,
            extractions=example_extractions
        )
    ]
    # 3 创建 LangExtract 模型
    # 4 执行 LangExtract 提取
    extract_result = lx.extract(text_or_documents=input_text,
                                prompt_description=langextract_prompt,
                                examples=examples,
                                model=deepseek_model,
                                extraction_passes=2,  # 多轮提取提高召回率;长文本时有效
                                max_workers=20,  # 并行处理加速;长文本时有效
                                max_char_buffer=1000,  # 较小的上下文提高准确性;长文本时有效
                                fence_output=True,  # 要求 LLM 输出用代码块包裹，避免格式错误
                                use_schema_constraints=True,  # 使用严格的 schema 约束，XX(提高灵活性)
                                prompt_validation_level=PromptValidationLevel.OFF,  # 关闭提示词验证
                                show_progress=True,  # 显示提取进度
                                )
    # 保存提取结果为 JSONL 格式
    lx.io.save_annotated_documents(
        extract_result,
        output_name=f"{pdf_path.stem}_LangExtract.json",
        output_dir=str(OUTPUT_DIR)
    )
    print(f"结果已保存: {OUTPUT_DIR}/{pdf_path.stem}_LangExtract.json")

    # 打印 LangExtract 结果
    print("LangExtract 提取结果:")
    print("=" * 80)
    if isinstance(extract_result, list):
        for extract_buffer in extract_result:
            for ext in extract_buffer.extractions:
                if ext.char_interval:
                    pos_info = f"[{ext.char_interval.start_pos}-{ext.char_interval.end_pos}]"
                else:
                    pos_info = f"[~]"
                print(f"{ext.extraction_index}: [{ext.extraction_class}] {ext.extraction_text} {pos_info}")
            print("=" * 80)
    elif isinstance(extract_result, lx.data.AnnotatedDocument):
        for ext in extract_result.extractions:
            if ext.char_interval:
                pos_info = f"[{ext.char_interval.start_pos}-{ext.char_interval.end_pos}]"
            else:
                pos_info = f"[~]"
            print(f"{ext.extraction_index}: [{ext.extraction_class}] {ext.extraction_text} {pos_info}")
        print("=" * 80)
