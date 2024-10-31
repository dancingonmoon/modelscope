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
    photo_path = r"C:/Users/shoub/Pictures\ticke2.jpg"
    model = genai.GenerativeModel("gemini-1.5-flash")
    photo = PIL.Image.open(photo_path)
    # response = model.generate_content(["请将图片文字提取,按原格式转成markdown格式", photo])
    # print(response.text)

    # Upload the file and print a confirmation
    pdf_file = genai.upload_file(path=r"L:/temp/刘禹/Испытания пластины редакция сж.pdf", display_name="Bullet.pdf")

    print(f"Uploaded file '{pdf_file.display_name}' as: {pdf_file.uri}")

        # Prompt the model with text and the previously uploaded image.
    response = model.generate_content([pdf_file, "请将PDF按照原格式翻译成中文,并按照原文档结构输出成Markdown格式"])

    print(response.text)