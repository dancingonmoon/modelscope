# -*- coding: utf-8 -*-

import requests
import random
from hashlib import md5
import pandas as pd
import openpyxl
import configparser


# https://fanyi-api.baidu.com/api/trans/product/desktop?req=developer

def config_read(config_path, section='DingTalkAPP_chatGLM', option1='Client_ID', option2=None):
    """
    option2 = None 时,仅输出第一个option1的值; 否则输出section下的option1与option2两个值
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    option1_value = config.get(section=section, option=option1)
    if option2 is not None:
        option2_value = config.get(section=section, option=option2)
        return option1_value, option2_value
    else:
        return option1_value


# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


# 定义百度翻译ＡＰＩ函数
def BaiduTranslateAPI(text, appid, appkey, from_lang='auto', to_lang='en'):
    """
    appid: 百度openAPI appid
    appkey: 百度openAPI appkey
    For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21
    """
    salt = random.randint(32768, 65536)
    s = ''.join([appid, text, str(salt), appkey])
    sign = make_md5(s)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': text, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = ''.join([endpoint, path])
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    try:
        trans_result = result['trans_result'][0]['dst']
        return trans_result
    except Exception as e:
        return result

    # 第一个方括号为字典的指定key取值,第二个方括号为之前取出的值为列表,列表的第一个元素,
    # 第三个方括号为取出的第一个元素继续为字典,取字典的值


# 定义函数:DataFrame单列遍历,调用百度翻译API,并生成翻译后的DataFrame

def DFColumnTranslate(DFColumn, appid, appkey, from_lang='auto', to_lang='en'):
    """
    百度翻译API下的单列文本翻译, 输入DataFrame列,翻译后,返回DataFrame列,包括列名翻译;空白文本自动跳过;
    DFColum: DataFrame单列
    from_lang: 源语言
    to_lang:目标语言
    """
    Translate = []
    for value in DFColumn.values:
        if value[0] != '':
            Translate.append(BaiduTranslateAPI(str(value[0]), appid, appkey, from_lang, to_lang))
        else:
            Translate.append(value)
    Column_name = DFColumn.keys()[0]
    if isinstance(Column_name, str):
        Column_name = BaiduTranslateAPI(Column_name, appid, appkey, from_lang, to_lang)

    Translate = pd.DataFrame(Translate, columns=[Column_name])
    return Translate


def Write2Excel(DFColumn, Write2Path, Write2Sheet, Write2Row=0, Write2Col=0, Writeheader=True):
    """
    将DataFrame追加写入已经存在的Excel,写入指定的sheet, 从指定的行列号开始写完整的DataFrame块;
    DataFrame: 写入的DataFrame
    Write2Path: 已经存在的Excel的完整路径;
    Write2Sheet: 待写入的Excel的sheet名;可以是新sheet;
    Write2Row: 写入的起始行,默认为0;
    Write2Col: 写入的起始列,默认为0,
    Writeheader: bool类型, 是否写入标题行,默认为True;
    """
    with pd.ExcelWriter(Write2Path, mode='a', engine="openpyxl", if_sheet_exists='overlay') as writer:
        DFColumn.to_excel(writer, Write2Sheet, startrow=Write2Row, startcol=Write2Col, index=False, header=Writeheader)
        # writer.save()
        # writer.close()


def ExcelColumnTranslate(appid, appkey, from_lang='auto', to_lang='en', ExcelPath: str = None,
                         readSheet=None, readHeader: int = None, readStartRow: int = 1, readCol=None, nrows: int = None,
                         write2Sheet: str = None, write2Row: int = 0, write2Col: int = 1):
    """
    Excel中读取单列,或者block,并调用百度翻译API,翻译后,写入已经存在的Excel,写入指定的sheet, 从指定的行列号追加写入完整的列或者块;
    appid: 百度openAPI appid
    appkey: 百度openAPI appkey
    from_lang: 源语言
    to_lang: 目标语言(For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21)
    ExcelPath: 待读取,翻译以及写入的Excel的完整路径;
    readSheet: 待读取的sheet name;
    readHeader: 待读取标题行行号(Excel的行号,从1开始计数); 缺省None
    readStartRow: 待读取的起始行(Excel的行号,从1开始计数); 缺省1;
        当readHeader不为None时,readStartRow=None表示从readHeader以下开始
    readCol: 待读取的起始列号;
    nrows: 待读取行的数量; 初始值None,表示读取整列;
    write2Sheet: 待写入的Excel的sheet名;可以是新sheet;
    write2Row: 写入的起始行,(不包括标题列);默认为1;
    write2Col: 写入的起始列,默认为0,
    """
    if readHeader is not None:
        if readStartRow is None:
            skiprows = None
        else:
            if readStartRow > readHeader:
                skiprows = range(readHeader, readStartRow - 1)
            else:
                skiprows = None

        readHeader -= 1  # 因为读取的标题行行号是从1开始计数的,所以要减1,变成index
        writeheader = True
        write2Row -= 2  # 带标题,至少从第二行开始写入,因为第一行要留标题
    else:
        skiprows = range(0, readStartRow - 1)
        writeheader = False
        write2Row -= 1

    ExcelData = pd.read_excel(ExcelPath, readSheet, header=readHeader, skiprows=skiprows, usecols=[readCol],
                              nrows=nrows, na_values='')
    TransData = DFColumnTranslate(ExcelData, appid, appkey, from_lang, to_lang)

    Write2Excel(TransData, ExcelPath, write2Sheet, write2Row, write2Col, writeheader)  # -2 因为index与行号差1


if __name__ == "__main__":
    # 读取需要翻译的Excel文件,装入DataFrame
    from_lang = 'auto'
    to_lang = 'en'  # 中文: zh;文言文: wyw;日本: jp; --> 伊朗语: ir; 波斯语: per
    # ExcelDirectory = r"E:/Working Documents/Eastcom/Russia/Igor/专网/LeoTelecom/发货测试验收/"
    ExcelPath = r"L:/temp/baiduTranslateTest.xlsx"
    readSheet = 'Sheet1'
    readHeader = 3
    readStartRow = None
    readCol = 1  # 也可以用列表读取多列
    nrows = 5
    write2Sheet = 'Sheet3'
    write2Row = 1
    write2Col = 1

    config_path = r"L:/Python_WorkSpace/config/baidu_OpenAPI.ini"

    appid, appkey = config_read(config_path, section="baidu_OpenAPI", option1='appid', option2='appkey')

    # if readHeader is not None:
    #     if readStartRow is None:
    #         skiprows = None
    #     else:
    #         if readStartRow > readHeader:
    #             skiprows = range(readHeader, readStartRow-1)
    #         else:
    #             skiprows = None
    # else:
    #     skiprows = range(1, readStartRow)
    #
    # ExcelData = pd.read_excel(ExcelPath, readSheet, header=readHeader - 1, skiprows=skiprows, usecols=[readCol],
    #                           nrows=nrows, na_values='')
    # pass

    ExcelColumnTranslate(appid=appid, appkey=appkey, from_lang=from_lang, to_lang=to_lang, ExcelPath=ExcelPath,
                         readSheet=readSheet, readHeader=readHeader, readStartRow=readStartRow, readCol=readCol,
                         nrows=nrows,
                         write2Sheet=write2Sheet, write2Row=write2Row, write2Col=write2Col)
    # ExcelData = pd.read_excel(''.join([ExcelDirectory, ExcelName]), ExcelSheet, header=Read_RowHeader,
    #                           usecols=Read_Column, nrows=10)
    # # 先处理缺失值,填充为''
    # ExcelData.fillna('', inplace=True)
    # # ExcelData[ExcelData.keys()[2]]

    # 单列DataFrame翻译结果,写入已经存在的Excel文件中,指定的Sheet表的指定单元格
    # WriteDirectory = ExcelDirectory
    # # WriteExcelName = '个人报销 -翻译测试.xlsx'
    # WriteExcelName = ExcelName
    # # WriteSheetName = '20200831'
    # WriteSheetName = 'Sheet3'
    # StartRow = 2
    # Write2Cols = [7,9]

    # 多列循环调用翻译结果写入函数,写入
    # TranslateDF = pd.DataFrame()
    # for i in range(0, 1):
    #     temp = ColumnTranslate(ExcelData[ExcelData.keys()[i]])
    #     TranslateDF[temp.keys()] = temp
    #     # print(temp.keys()[0])
    #     WriteColumnTranslate(TranslateDF, WriteDirectory, WriteExcelName, WriteSheetName, StartRow, StartCol)
    # TranslateDF

    # with pd.ExcelWriter(WriteDirectory + WriteExcelName, mode='a', engine="openpyxl",
    #                     if_sheet_exists="overlay") as writer:
    #     # Workbook = openpyxl.load_workbook(WriteDirectory + WriteExcelName)  # 读取要写入的workbook
    #     # writer.book = Workbook  # 此时的writer里还只是读写器. 然后将上面读取的Workbook复制给writer
    #     # writer.sheets = dict((ws.title, ws) for ws in Workbook.worksheets)  # 复制存在的表
    #     ExcelData.to_excel(writer, WriteSheetName, startrow=StartRow,
    #                        startcol=StartCol, index=False, )
    # writer.save()
    # writer.close()
    # Write2Excel(ExcelData, ''.join([WriteDirectory, WriteExcelName]), WriteSheetName, StartRow, Write2Cols=Write2Cols)
