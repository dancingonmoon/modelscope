# -*- coding: utf-8 -*-
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import requests



def get_signed_url(
        endpoint:str="https://oss-cn-hangzhou.aliyuncs.com",
        region:str="cn-hangzhou",
        BucketName:str=None,
        object_name:str=None,
        expiration=600,
        accelerate:bool=False,
        accelreate_endpoint:str="https://oss-accelerate.aliyuncs.com"):
    """
    签名URL是一种由客户端基于本地密钥信息生成的临时授权访问链接，可允许第三方在有效期内下载或预览私有文件，而无需暴露访问密钥.
    生成签名URL过程中，SDK利用本地存储的密钥信息，根据特定算法计算出签名（signature），然后将其附加到URL上，以确保URL的有效性和安全性。
    通过命令行工具ossutil和SDK生成的签名URL，最大有效时长为7天。
    :param endpoint:  填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
    :param region:  填写Endpoint对应的Region信息，例如cn-hangzhou。注意，v4签名下，必须填写该参数
    :param BucketName:  yourBucketName填写存储空间名称
    :param object_name: 填写Object完整路径，例如exampledir/exampleobject.txt。Object完整路径中不能包含Bucket名称
    :return:
    """
    # 从环境变量中获取访问凭证。运行本代码示例之前，请确保已设置环境变量OSS_ACCESS_KEY_ID和OSS_ACCESS_KEY_SECRET。
    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    # yourBucketName填写存储空间名称。
    bucket = oss2.Bucket(auth, endpoint, BucketName, region=region)


    # 生成下载文件的签名URL，有效时间为600秒。
    # 生成签名URL时，OSS默认会对Object完整路径中的正斜线（/）进行转义，从而导致生成的签名URL无法直接使用。
    # 设置slash_safe为True，OSS不会对Object完整路径中的正斜线（/）进行转义，此时生成的签名URL可以直接使用。
    if accelerate:
        endpoint = accelreate_endpoint
    url = bucket.sign_url('GET', object_name, expires=expiration, slash_safe=True)
    # print('签名URL的地址为：', url)
    return url

def get_object(
        file_url:str=None,
        save_path:str="d:/downloads/myfile.txt",
        ):
    """
    使用requests库下载文件
    :param bucket_name:
    :param object_name:
    :return:
    """
    try:
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(4096):
                    f.write(chunk)
            print("Download completed!")
        else:
            print(f"No file to download. Server replied HTTP code: {response.status_code}")
    except Exception as e:
        print("Error during download:", e)