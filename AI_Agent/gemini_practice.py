import google.generativeai as genai
import os
from GLM.GLM_callFunc import config_read
import PIL.Image



if __name__ == "__main__":
    config_path_gemini = r"l:/Python_WorkSpace/config/geminiAPI.ini"
    config_path_zhipuai = r"l:/Python_WorkSpace/config/zhipuai_SDK.ini"

    # gemini API
    geminiAPI = config_read( config_path_gemini, section="gemini_API", option1="api_key" )
    genai.configure(api_key=geminiAPI)

    # JPG filepath:
    photo_path = r"C:/Users/shoub/Pictures/融合通信架构.png"
    model = genai.GenerativeModel('gemini-1.5-pro')
    # photo = PIL.Image.open(photo_path)
    # response = model.generate_content(["请将图片文字提取,按原格式转成markdown格式", photo])
    # print(response.text)

    # Upload the file and print a confirmation
    # pdf_file = genai.upload_file(path=r"L:/temp/刘禹/Испытания пластины редакция сж.pdf", display_name="Bullet.pdf")
    # pdf_file = genai.upload_file(path=photo_path, display_name="融合通信架构.pdf")

    # print(f"Uploaded file '{pdf_file.display_name}' as: {pdf_file.uri}")

    # Configure a model to use Google Search : Grounding is available to test for free in Google AI Studio.
    # In the API, developers can access the tool with the paid tier for $35 per 1,000 grounded queries.
    # response = model.generate_content(
    #                                     contents="请总结今天中国新闻的头条,并以markdown格式输出",
    #                                     tools={"google_search_retrieval": {
    #                                         "dynamic_retrieval_config": {
    #                                             "mode": "unspecified",
    #                                             "dynamic_threshold": 0.3}}}
    #                                         )
    # Prompt the model with text and the previously uploaded image.
    # response = model.generate_content([pdf_file, "请将图片按照原格式翻译成中文,并按照原文档结构,布局,输出成Markdown格式"])

    # print(response.text)

    # 聊天流式传输，直播聊天功能：
    chat = model.start_chat(
        history=None, )
    for i in range(10):
        message = input("请输入提示词：")
        response = chat.send_message(message, stream=True)
        for chunk in response:
            print(chunk.text)
            print("_" * 80)
        print(chat.history)

