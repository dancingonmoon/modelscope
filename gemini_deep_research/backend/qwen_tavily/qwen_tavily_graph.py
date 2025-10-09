import os
# import sys
# # 获取当前文件所在目录，然后添加正确的父级目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.join(current_dir, '..')
# sys.path.append(os.path.abspath(current_dir))
# sys.path.append(os.path.abspath(parent_dir))
# print(f"graph.py, current_dir:{current_dir}")

from ..tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch

from ..state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from configuration import Configuration
from ..prompts_cn import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_qwq import ChatQwen
from ..utils import (
    get_research_topic,
)

load_dotenv(override=True)

if os.getenv("DASHSCOPE_API_KEY") is None:
    raise ValueError("DASHSCOPE_API_KEY is not set")

if os.getenv("TAVILY_API_KEY") is None:
    raise ValueError("TAVILY_API_KEY is not set")

# Initialize Tavily Search
tavily_search = TavilySearch(
    max_results=5,
    search_depth="advanced",
    api_key=os.getenv("TAVILY_API_KEY")
)


# Nodes
async def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Qwen to create optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init DeepSeek
    llm = ChatQwen(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = await structured_llm.ainvoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


async def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using Tavily Search API.

    Executes a web search using Tavily Search API and then uses DeepSeek to analyze and summarize the results.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    
    # Perform search using Tavily
    search_results = await tavily_search.ainvoke(state["search_query"])
    
    # Extract content and URLs from search results
    search_content = ""
    sources_gathered = []
    

    
    # Handle different return formats from Tavily
    if isinstance(search_results, list):
        results_to_process = search_results
    elif isinstance(search_results, dict):
        # Tavily typically returns a dict with 'results' key
        results_to_process = search_results.get('results', [])
    elif isinstance(search_results, str):
        # If it's a string, it might be JSON content
        try:
            import json
            parsed_results = json.loads(search_results)
            if isinstance(parsed_results, dict) and 'results' in parsed_results:
                results_to_process = parsed_results['results']
            else:
                results_to_process = [{"title": "Search Result", "url": "", "content": search_results}]
        except:
            results_to_process = [{"title": "Search Result", "url": "", "content": search_results}]
    else:
        results_to_process = []
    

    
    for i, result in enumerate(results_to_process):
        if isinstance(result, dict):
            title = result.get('title', f'Result {i+1}')
            url = result.get('url', f'https://search-result-{i+1}.com')
            content = result.get('content', str(result))
        else:
            title = f'Result {i+1}'
            url = f'https://search-result-{i+1}.com'
            content = str(result)
            
        search_content += f"Source {i+1}: {title}\nURL: {url}\nContent: {content}\n\n"
        sources_gathered.append({
            "title": title,
            "url": url,
            "content": content[:500] + "..." if len(content) > 500 else content,
            "short_url": f"[{i+1}]",
            "value": url,
            "label": title  # Add label field for frontend compatibility
        })
    

    
    # Format prompt for DeepSeek to analyze the search results
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )
    
    # Add search results to the prompt
    analysis_prompt = f"{formatted_prompt}\n\n搜索结果：\n{search_content}\n\n请分析这些搜索结果并提供带有引用的综合摘要。请用中文回答。"
    
    # Use DeepSeek to analyze and summarize the search results
    llm = ChatQwen(
        model=configurable.query_generator_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    
    response = await llm.ainvoke(analysis_prompt)
    
    # Insert citation markers
    modified_text = response.content
    for i, source in enumerate(sources_gathered):
        # Replace URL references with short citations
        if source['url'] in modified_text:
            modified_text = modified_text.replace(source['url'], source['short_url'])
        # Also try to match domain names
        domain = source['url'].split('/')[2] if len(source['url'].split('/')) > 2 else source['url']
        if domain in modified_text:
            modified_text = modified_text.replace(domain, source['short_url'])

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


async def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = ChatQwen(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    result = await llm.with_structured_output(Reflection).ainvoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


async def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model, default to DeepSeek
    llm = ChatQwen(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
    result = await llm.ainvoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, input=MessagesState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
