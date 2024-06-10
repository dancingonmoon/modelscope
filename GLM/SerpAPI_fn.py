# https://serpapi.com/search-api

from serpapi import GoogleSearch, BaiduSearch
import configparser


def config_read(config_path, section='Serp_API', option1='api_key', option2=None):
    """
    option2 = None 时,仅输出第一个option1的值; 否则输出section下的option1与option2两个值
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    option1_value = config.get(section=section, option=option1)
    if option2 is not None:
        option2_value = config.get(section=section, option=option2)
        return option1_value, option2_value
    else:
        return option1_value

def serpapi_GoogleSearch(api_key, query, location='Hong Kong', hl='zh-cn', gl='cn', tbs=None, tbm=None, num=30, ):
    """
    使用SerpAPI进行Google搜索
    args:
        api_key: SerpAPI的api_key
        query: 搜索的问题或关键字
        location: Parameter defines from where you want the search to originate.
        hl:Parameter defines the country to use for the Google search. It's a two-letter country code. (e.g., 'us' for the
            United States, uk for the United Kingdom, or 'fr' for France)
        gl:Parameter defines the language to use for the Google search. It's a two-letter language code. (e.g., en for
            English, es for Spanish, or fr for French). Head to the Google languages page for a full list of supported
            Google languages.
        num:Parameter defines the maximum number of results to return. (e.g., 10 (default) returns 10 results
        tbs:(to be searched) parameter defines advanced search parameters that aren't possible in the regular query
        field. (e.g., advanced search for patents, dates, news, videos, images, apps, or text contents).
        tbm: (to be matched) parameter defines the type of search you want to do.
            It can be set to:
            (no tbm parameter): regular Google Search,
            isch: Google Images API,
            lcl - Google Local API
            vid: Google Videos API,
            nws: Google News API,
            shop: Google Shopping API,
            pts: Google Patents API,
            or any other Google service.

    out:
        result: a structured JSON of the google search results
    """
    param = {
        "q": query,
        "location": location,
        "api_key": api_key,
        "hl": hl,
        "gl": gl,
        "num": num,
        "tbm": tbm,
        "tbs": tbs
    }
    search = GoogleSearch(param)
    result = search.get_dict()
    return result


def serpapi_BaiduSearch(api_key, query, ct=1, rn=50, engine='Baidu'):
    """
    使用SerpAPI进行Baidu搜索
    args:
        api_key: SerpAPI的API key
        query: 搜索的问题或关键字
        ct: Parameter defines which language to restrict results. Available options:
            1 - All languages
            2 - Simplified Chinese
            3 - Traditional Chinese
        rn: Parameter defines the maximum number of results to return, limited to 50. (e.g., 10 (default) returns 10 results,
        engine: Set parameter to Baidu to use the Baidu API engine.

    out:
        result: a structured JSON of the baidu search results
    """
    param = {
        "q": query,
        'ct': ct,
        'rn': rn,
        'engine': engine,
        'api_key': api_key
    }
    search = BaiduSearch(param)
    result = search.get_dict()
    return result


if __name__ == "__main__":
    config_path = r"e:/Python_WorkSpace/config/SerpAPI.ini"
    api_key = config_read(config_path, section='Serp_API', option1='api_key')
    query = 'Tucker Carson与普京的会面,都谈了些什么?'

    # search_result = serpapi_GoogleSearch(api_key, query, location='Hong Kong',
    #                                      hl='zh-cn',
    #                                      gl='cn',
    #                                      tbs='News', tbm=None,
    #                                      num=30)
    search_result = serpapi_BaiduSearch(api_key, query, ct=2, rn=30)
    print(search_result)
