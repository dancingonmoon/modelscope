"""
Qwen-Deep-Research 是通义千问的研究型智能体模型。它可拆解复杂问题，结合互联网搜索进行推理分析并生成研究报告。
上下文长度 1,000,000
最大输入 997,952
最大输出 32,768
输入成本 0.054元/每千Token
输出成本 0.163元/每千Token
免费额度 无免费额度
以下代码旨在使用模型研究人工智能在教育领域中的应用，具体细分为个性化学习和智能评估两个方面。
"""
import os
import dashscope
from pathlib import Path

# 配置API Key
# 若没有配置环境变量，请用百炼API Key将下行替换为：API_KEY = "sk-xxx"
API_KEY = os.getenv('DASHSCOPE_API_KEY')


def call_deep_research_model(messages, step_name):
    print(f"\n=== {step_name} ===")

    try:
        responses = dashscope.Generation.call(
            api_key=API_KEY,
            model="qwen-deep-research",
            messages=messages,
            # qwen-deep-research模型目前仅支持流式输出
            stream=True
            # incremental_output=True 使用增量输出请添加此参数
        )

        return process_responses(responses, step_name)

    except Exception as e:
        print(f"调用API时发生错误: {e}")
        return ""


# 显示阶段内容
def display_phase_content(phase, content, status):
    if content:
        print(f"\n[{phase}] {status}: {content}")
    else:
        print(f"\n[{phase}] {status}")


# 处理响应
def process_responses(responses, step_name):
    current_phase = None
    phase_content = ""
    research_goal = ""
    web_sites = []
    keepalive_shown = False  # 标记是否已经显示过KeepAlive提示

    for response in responses:
        # 检查响应状态码
        if hasattr(response, 'status_code') and response.status_code != 200:
            print(f"HTTP返回码：{response.status_code}")
            if hasattr(response, 'code'):
                print(f"错误码：{response.code}")
            if hasattr(response, 'message'):
                print(f"错误信息：{response.message}")
            print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
            continue

        if hasattr(response, 'output') and response.output:
            message = response.output.get('message', {})
            phase = message.get('phase')
            content = message.get('content', '')
            status = message.get('status')
            extra = message.get('extra', {})

            # 阶段变化检测
            if phase != current_phase:
                if current_phase and phase_content:
                    # 根据阶段名称和步骤名称来显示不同的完成描述
                    if step_name == "第一步：模型反问确认" and current_phase == "answer":
                        print(f"\n 模型反问阶段完成")
                    else:
                        print(f"\n {current_phase} 阶段完成")
                current_phase = phase
                phase_content = ""
                keepalive_shown = False  # 重置KeepAlive提示标记

                # 根据阶段名称和步骤名称来显示不同的描述
                if step_name == "第一步：模型反问确认" and phase == "answer":
                    print(f"\n 进入模型反问阶段")
                else:
                    print(f"\n 进入 {phase} 阶段")

            # 处理WebResearch阶段的特殊信息
            if phase == "WebResearch":
                if extra.get('deep_research', {}).get('research'):
                    research_info = extra['deep_research']['research']

                    # 处理streamingQueries状态
                    if status == "streamingQueries":
                        if 'researchGoal' in research_info:
                            goal = research_info['researchGoal']
                            if goal:
                                research_goal += goal
                                print(f"\n   研究目标: {goal}", end='', flush=True)

                    # 处理streamingWebResult状态
                    elif status == "streamingWebResult":
                        if 'webSites' in research_info:
                            sites = research_info['webSites']
                            if sites and sites != web_sites:  # 避免重复显示
                                web_sites = sites
                                print(f"\n   找到 {len(sites)} 个相关网站:")
                                for i, site in enumerate(sites, 1):
                                    print(f"     {i}. {site.get('title', '无标题')}")
                                    print(f"        描述: {site.get('description', '无描述')[:100]}...")
                                    print(f"        URL: {site.get('url', '无链接')}")
                                    if site.get('favicon'):
                                        print(f"        图标: {site['favicon']}")
                                    print()

                    # 处理WebResultFinished状态
                    elif status == "WebResultFinished":
                        print(f"\n   网络搜索完成，共找到 {len(web_sites)} 个参考信息源")
                        if research_goal:
                            print(f"   研究目标: {research_goal}")

            # 累积内容并显示
            if content:
                phase_content += content
                # 实时显示内容
                print(content, end='', flush=True)

            # 显示阶段状态变化
            if status and status != "typing":
                print(f"\n   状态: {status}")

                # 显示状态说明
                if status == "streamingQueries":
                    print("   → 正在生成研究目标和搜索查询（WebResearch阶段）")
                elif status == "streamingWebResult":
                    print("   → 正在执行搜索、网页阅读和代码执行（WebResearch阶段）")
                elif status == "WebResultFinished":
                    print("   → 网络搜索阶段完成（WebResearch阶段）")

            # 当状态为finished时，显示token消耗情况
            if status == "finished":
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    print(f"\n    Token消耗统计:")
                    print(f"      输入tokens: {usage.get('input_tokens', 0)}")
                    print(f"      输出tokens: {usage.get('output_tokens', 0)}")
                    print(f"      请求ID: {response.get('request_id', '未知')}")

            if phase == "KeepAlive":
                # 只在第一次进入KeepAlive阶段时显示提示
                if not keepalive_shown:
                    print("当前步骤已经完成，准备开始下一步骤工作")
                    keepalive_shown = True
                continue

    if current_phase and phase_content:
        if step_name == "第一步：模型反问确认" and current_phase == "answer":
            print(f"\n 模型反问阶段完成")
        else:
            print(f"\n {current_phase} 阶段完成")

    return phase_content


def main(subject: str, output_filepath: Path|str = None):
    # 检查API Key
    if not API_KEY:
        print("错误：未设置 DASHSCOPE_API_KEY 环境变量")
        print("请设置环境变量或直接在代码中修改 API_KEY 变量")
        return

    # print("用户发起对话：研究一下人工智能在教育中的应用")
    print(f"用户发起对话：{subject}")

    # 第一步：模型反问确认
    # 模型会分析用户问题，提出细化问题来明确研究方向
    start_message = {'role': 'user', 'content': subject}
    messages = [start_message]
    step1_content = call_deep_research_model(messages, "第一步：模型反问确认")

    # 第二步：深入研究
    # 基于第一步的反问内容，模型会执行完整的研究流程
    messages = [
        # {'role': 'user', 'content': '研究一下人工智能在教育中的应用'},
        start_message,
        {'role': 'assistant', 'content': step1_content},  # 包含模型的反问内容
        {'role': 'user', 'content': '我主要关注个性化学习和智能评估这两个方面'}
    ]

    step2_content = call_deep_research_model(messages, "第二步：深入研究")
    print("\n 研究完成！")
    if output_filepath:
        if isinstance(output_filepath, str):
            output_filepath = Path(output_filepath)
        with open(output_filepath, 'a', encoding='utf-8') as f:
            f.write('--------------------')
            f.write(step2_content)




if __name__ == "__main__":
    subject = """
              TETRA制式的集群系统中,规范中定义了多个频段，但是全球各个国家频段使用情况却不一样；
              请分析下全球TETRA在450M-470M这个频段的应用情况研究。研究报告的读者主要是产品研发的决策者，需要决策以下：
              目前的TETRA产品线已经有350M，380M，410M,800M四个频段，唯独缺失450M这个频段。俄罗斯的客户现在有450M TETRA的需求。
              据客户反馈：俄罗斯目前国内TETRA允许频段有410M，450M两个频段；尽管我们的410M TETRA产品已经在俄罗斯的油气田项目
              获得使用，然而，莫斯科的两大机场目前允许的频率缺失450M；并且客户反馈，由于410M这个频段在俄罗斯某些地区被电力
              系统使用，故而TETRA系统只能使用450M这个频段。客户还反馈，俄罗斯的欧洲部分使用450M，而俄罗斯的亚洲部分却主要使用410M，
              尽管450M频段的TETRA俄罗斯全境都可以使用，但410M的TETRA市场占有率仍然更高。
              请分析以上客户反馈情况是否属实，然后，介绍下俄罗斯450M TETRA市场情况，都在哪些地区使用，存量市场占有率如何？
              然后，再分析未来3-5年，450M TETRA的市场容量预测，项目预测。
              接着，再从前苏联地区450M TETRA存量市场角度予以分析，450M市场情况，以及未来3-5年450M的容量预测，项目预测；
              最后，再从全球国家地区角度，分析下450M存量市场占有情况，以及未来3-5年450M容量分析，项目预测。
              以上分析，需要有数据支撑，以便我们的产品部门能够决策，是否需要在已有的产品线基础上再开发450M基站产品线。
              """
    output_filepath = r"E:/Working Documents/Eastcom/产品/无集/多网融合/450M TETRA/450M TETRA Research.txt"
    main(subject=subject, output_filepath=output_filepath)
