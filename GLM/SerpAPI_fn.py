# https://serpapi.com/search-api

from serpapi import GoogleSearch, BaiduSearch
import configparser


def get_api_key(config_file_path, section='Serp_API', option='api_key'):
    """
    从配置文件config.ini中,读取api_key,api_secret;避免程序代码中明文显示key,secret.
    args:
        config_file_path: config.ini的文件路径(包含文件名,即: directory/config.ini)
        section: config.ini中的section名称;
        option: config.ini中的option名称;
    out:
        返回option对应的value值;此处为api_key
    """
    config = configparser.ConfigParser()
    config.read(config_file_path, encoding="utf-8")  # utf-8支持中文
    return config[section][option]


def serpapi_GoogleSearch(config_path, query, section='Serp_API', option='api_key',
                         location='Hong Kong', hl='zh-cn', gl='cn', tbs=None, tbm=None, num=30, ):
    """
    使用SerpAPI进行Google搜索
    args:
        config_path: config.ini的文件路径(包含文件名,即: directory/config.ini)
        section: config.ini中section名称;
        option: config.ini中option名称;
        query: 搜索的问题或关键字
        location: Parameter defines from where you want the search to originate.
        hl:Parameter defines the country to use for the Google search. It's a two-letter country code. (e.g., us for the
            United States, uk for United Kingdom, or fr for France)
        gl:Parameter defines the language to use for the Google search. It's a two-letter language code. (e.g., en for
            English, es for Spanish, or fr for French). Head to the Google languages page for a full list of supported
            Google languages.
        num:Parameter defines the maximum number of results to return. (e.g., 10 (default) returns 10 results
        tbs:(to be searched) parameter defines advanced search parameters that aren't possible in the regular query
        field. (e.g., advanced search for patents, dates, news, videos, images, apps, or text contents).
        tbm:(to be matched) parameter defines the type of search you want to do.
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
    api_key = get_api_key(config_path, section, option)
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


def serpapi_BaiduSearch(config_path, query, section='Serp_API', option='api_key',
                        ct=2, rn=50, engine='Baidu'):
    """
    使用SerpAPI进行Baidu搜索
    args:
        config_path: config.ini的文件路径(包含文件名,即: directory/config.ini)
        section: config.ini中section名称;
        option: config.ini中option名称;
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
    api_key = get_api_key(config_path, section, option)
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
    config_path = r"l:/Python_WorkSpace/config/SerpAPI.ini"
    query = 'Tucker Carson与普京的会面,都谈了些什么?'

    # search_result = serpapi_GoogleSearch(config_path, query, section='Serp_API', option='api_key', location='Hong Kong',
    #                                      hl='zh-cn',
    #                                      gl='cn',
    #                                      tbs='News', tbm=None,
    #                                      num=30)
    search_result = serpapi_BaiduSearch(config_path, query, section='Serp_API', option='api_key',
                                        ct=2, rn=30)
    print(search_result)
