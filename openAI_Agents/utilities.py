import os
import requests
from pathlib import Path
from datetime import datetime, timedelta

# https://help.aliyun.com/zh/model-studio/get-temporary-file-url?spm=a2c4g.11186623.help-menu-2400256.d_2_12_6.57eeb41fgvmxc2
def get_upload_policy(api_key, model_name):
    """
    获取文件上传凭证
    :param api_key:
    :param model_name: 例如："qwen-vl-plus"
    :return:
    """
    if not api_key:
        api_key=os.getenv("DASHSCOPE_API_KEY")

    url = "https://dashscope.aliyuncs.com/api/v1/uploads"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    parms = {
        "action": "getPolicy",
        "model": model_name
    }
    response = requests.get(url, headers=headers, params=parms)
    if response.status_code != 200:
        raise Exception(f"Failed to get upload policy: {response.status_code} - {response.text}")

    return response.json()["data"]


def upload_file_to_oss(policy_data, file_path):
    """
    将文件上传到临时存储OSS
    :param policy_data:
    :param file_path:
    :return:
    """
    file_name = Path(file_path).name
    key = f"{policy_data['upload_dir']}/{file_name}"

    with open(file_path, 'rb') as file:
        files = {
            "OSSAccessKeyId": (None, policy_data['oss_access_key_id']),
            "Signature": (None, policy_data['signature']),
            "policy": (None, policy_data['policy']),
            "x-oss-object-acl": (None, policy_data['x_oss_object_acl']),
            "x-oss-forbid-overwrite": (None, policy_data['x_oss_forbid_overwrite']),
            "key": (None, key),
            "success_action_status": (None, '200'),
            "file": (file_name, file)
        }

        response = requests.post(policy_data['upload_host'], files=files)
        if response.status_code != 200:
            raise Exception(f"Failed to upload file to OSS: {response.status_code} - {response.text}")
    return f"oss://{key}"  # 例如：oss://dashscope-instant/f81efcc38b72f8031fdf8b1618b70dc1/2025-06-02/b9d25c60-f8aa-9fc0-9b88-741123e74cf3/碑文.jpg


def upload_file_and_get_url(api_key, model_name, file_path):
    """
    上传文件并获取公网URL,文件URL仍在上传后的48小时有效期内。
    模型调用与文件上传凭证中使用的模型必须保持一致。
    模型调用的API KEY需与文件上传凭证中的API KEY同属一个阿里云主账号，不可使用其他账号的API KEY调用模型。
    在调用模型时，若使用临时存储空间中的文件，必须在 Header 中添加参数 X-DashScope-OssResourceResolve: enable，否则将报错。
    :param api_key:
    :param model_name:
    :param file_path:
    :return: oss://dashscope-instant/??????/file_path  (仅在阿里内部访问)
    """
    # 1. 获取上传凭证
    policy_data = get_upload_policy(api_key, model_name)
    #  2. 上传文件到OSS
    oss_url = upload_file_to_oss(policy_data, file_path)

    return oss_url


if __name__ == "__main__":
    # 从环境变量中获取API Key 或者 在代码中设置 api_key = "your_api_key"
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise Exception("请设置DASHSCOPE_API_KEY环境变量")

    # 设置model名称
    model_name = "qwen-vl-plus"

    # 待上传的文件路径
    file_path = r"C:\Users\shoub\Pictures\碑文.jpg"  # 替换为实际文件路径

    try:
        public_url = upload_file_and_get_url(api_key, model_name, file_path)
        expire_time = datetime.now() + timedelta(hours=48)
        print(f"文件上传成功，有效期为48小时，过期时间: {expire_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"公网URL: {public_url}")

    except Exception as e:
        print(f"Error: {str(e)}")