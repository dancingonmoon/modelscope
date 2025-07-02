# -*- coding: utf-8 -*-
import os
from typing import Literal
from dataclasses import dataclass
from openAI_Agents.openAI_Agents_practice import openAI_Agents_create, save2file, _Qwen_MT_func
from agents import Agent, Runner,TResponseInputItem, ItemHelpers
import asyncio
def qwen_VL():
    """
    openAI_Agents,llm为QwenVL,以及Qwen，根据prompt是否有图片，是否使用QwenVL,从而执行图片理解，并带有本地存储工具；
    :return: openAI_Agents_create 类，其中包含了agent等属性与方法
    """
    QwenVL_model = 'qwen-vl-plus-latest'
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QwenVL_agent_instruction = '''
        您是一个助人为乐的助手，可以根据传入的图片来进行:
        1)图像问答：描述图像中的内容或者对其进行分类打标，如识别人物、地点、花鸟鱼虫等。
        2)数学题目解答：解答图像中的数学问题，适用于中小学、大学以及成人教育阶段。
        3)视频理解：分析视频内容，如对具体事件进行定位并获取时间戳，或生成关键时间段的摘要。
        4)物体定位：定位图像中的物体，返回外边界矩形框的左上角、右下角坐标或者中心点坐标。
        5)文档解析：将图像类的文档（如扫描件/图片PDF）解析为 QwenVL HTML格式，该格式不仅能精准识别文本，还能获取图像、表格等元素的位置信息。
        6)文字识别与信息抽取：识别图像中的文字、公式，或者抽取票据、证件、表单中的信息，支持格式化输出文本；可识别的语言有中文、英语、日语、韩语、阿拉伯语、越南语、法语、德语、意大利语、西班牙语和俄语。
        你只对带有图片的prompt，做出响应。
        '''

    mcp_names = ['file_system']
    mcp_params = [{
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            ".",
        ]}]
    mcp_io_methods = ["MCPServerStdio"]

    QwenVL_agent = openAI_Agents_create(agent_name='通义千问视觉理解智能体',
                                        instruction=QwenVL_agent_instruction,
                                        model=QwenVL_model,
                                        base_url=None,
                                        api_key=None,
                                        tools=[save2file],
                                        handoff_description="当prompt有图片时,使用QwenVL模型进行视觉推理,并且必要时，按要求将约定的内容存入本地文件"
                                        )

    Qwen_model = 'qwen-turbo-latest'
    Qwen_model_instruction = """
        你是一名助人为乐的助手,
        1)当prompt中有文件时，请handoff至视觉推理模型;
        2)否则，就直接回答问题;
        3) 必要时，可以将约定的内容存入本地文件。
        """
    handoff_description = """
        本模型仅仅处理不带有文件的prompt;当prompt图片文件时，请handoff至视觉推理模型，并给出结果。
        """
    Qwen3_agent = openAI_Agents_create(agent_name='通义千问智能体(general)',
                                       instruction=Qwen_model_instruction,
                                       model=Qwen_model,
                                       base_url=None,
                                       api_key=None,
                                       tools=[save2file],
                                       handoffs=[QwenVL_agent.agent],
                                       handoff_description=handoff_description

                                       )
    # 运行主协程
    # await Qwen3_agent.chat_continuous(runner_mode='stream', enable_fileloading=True)
    # await QwenVL_agent.multi_mcp_chat_continuous(runner_mode='async', enable_fileloading=False,
    #                                                    mcp_names=mcp_names,
    #                                                    mcp_params=mcp_params,
    #                                                    mcp_io_methods=mcp_io_methods)
    return Qwen3_agent
@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]
def gemini_translate_agent():
    translate_model = "gemini-2.5-flash"
    evaluation_model = "gemini-1.5-pro"
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    extra_body = {
        'extra_body': {
            "google": {
                "thinking_config": {
                    "thinking_budget": 800,
                    "include_thoughts": True
                }
            }
        }
    }
    translate_agent_instruction = """
    你是一名翻译官，具备各类语言的文字，文档的翻译工作；并且根据原文的文体，原文谈及的领域，阅读对象，语气，使用合适的语言和文字，问题，语气来翻译，翻译结果专业，贴切。
    """
    evaluate_agent_instruction = """
    1.你是一个翻译评价家，根据翻译结果，判断翻译质量是否合格;
    2.如果不合格的话，你需要给出反馈意见，指明翻译需要改进的地方;
    3.评价的要求需要严格，尽量不要在首次评价中就给与翻译质量合格的决定。
    """

    translate_agent = openAI_Agents_create(
        agent_name='gemini2.5_flash_translator',
        instruction=translate_agent_instruction,
        model=translate_model,
        base_url=base_url,
        api_key=api_key,
        custom_extra_body=extra_body,
        tools=[save2file]
    )
    evaluate_agent = openAI_Agents_create(
        agent_name='gemini1.5_pro_evaluation',
        instruction=evaluate_agent_instruction,
        model=evaluation_model,
        base_url=base_url,
        api_key=api_key,
        output_type=EvaluationFeedback

    )


    return translate_agent, evaluate_agent


async def gemini_translator(translate_agent:Agent, evaluate_agent:Agent, input_items: list[TResponseInputItem] | TResponseInputItem):

    while True:
        # translate_result = await translate_agent.async_chat_once(input_items)
        translate_result = await Runner.run(translate_agent, input_items)

        input_items = translate_result.to_input_list()
        latest_outline = ItemHelpers.text_message_outputs(translate_result.new_items)
        print("translation generated")

        evaluator_result = await Runner.run(evaluate_agent, input_items)
        result: EvaluationFeedback = evaluator_result.final_output
        print(f"Evaluator score: {result.score}")

        if result.score == "pass":
            print("translation is good enough, exiting.")
            break

        print("Re-running with feedback")

        input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})

    print(f"Final translation: {latest_outline}")
    return latest_outline


if __name__ == '__main__':
    agent = qwen_VL()
    # asyncio.run(agent.chat_continuous(runner_mode='stream', enable_fileloading=True))

    translate_agent, evaluate_agent = gemini_translate_agent()
    msg = input("天下为公")
    input_items: list[TResponseInputItem] = [{"content": msg, "role": "user"}]
    asyncio.run(gemini_translator(translate_agent=translate_agent.agent,
                                  evaluate_agent=evaluate_agent.agent,
                                  input_items=input_items))