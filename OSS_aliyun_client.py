# -*- coding: utf-8 -*-
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import requests
from itertools import islice
import logging
import sys

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 当无法确定待上传的数据长度时，total_bytes的值为None。
def percentage_callback(consumed_bytes, total_bytes):
    """
    用于多线程,断点续传时的回调函数,打印上传进度
    :param consumed_bytes:
    :param total_bytes: 当无法确定待上传的数据长度时，total_bytes的值为None。
    :return:
    """
    if total_bytes:
        rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
        print('\r{0}% '.format(rate), end='')
        sys.stdout.flush()


class BucketObject:
    """
    实现bucket对象的签名,上传,下载,列举
    """

    def __init__(self,
                 endpoint: str = "https://oss-cn-hangzhou.aliyuncs.com",
                 region: str = "cn-hangzhou",
                 bucketname: str = None,
                 logger: logging.Logger = None
                 ):
        """
        :param endpoint:  填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
        :param region:  填写Endpoint对应的Region信息，例如cn-hangzhou。注意，v4签名下，必须填写该参数
        :param bucketname:  yourBucketName填写存储空间名称
        :return:
        """
        if logger is None:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        # 从环境变量中获取访问凭证。运行本代码示例之前，请确保已设置环境变量OSS_ACCESS_KEY_ID和OSS_ACCESS_KEY_SECRET。
        auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
        self.auth = auth
        # yourBucketName填写存储空间名称。
        self.endpoint = endpoint
        self.bucketname = bucketname
        self.region = region
        self.bucket = oss2.Bucket(auth, endpoint, bucketname, region=region)

    def get_signed_url(self,
                       object_name: str = None,
                       expiration: int = 600,
                       accelerate: bool = False,
                       accelreate_endpoint: str = "https://oss-accelerate.aliyuncs.com"):
        """
        签名URL是一种由客户端基于本地密钥信息生成的临时授权访问链接，可允许第三方在有效期内下载或预览私有文件，而无需暴露访问密钥.
        生成签名URL过程中，SDK利用本地存储的密钥信息，根据特定算法计算出签名（signature），然后将其附加到URL上，以确保URL的有效性和安全性。
        通过命令行工具ossutil和SDK生成的签名URL，最大有效时长为7天。
        :param object_name: 填写Object完整路径，例如exampledir/exampleobject.txt。Object完整路径中不能包含Bucket名称
        :param expiration: 生成签名URL的有效时长，单位为秒。默认值为600秒，即10分钟。最大有效时长为7天, 即7*24*3600秒;
        :param accelerate: 是否使用OSS传输加速, 每GB收费1.25元;
        :param accelreate_endpoint: 当使用OSS传输加速时,endpoint为加速域名, 如: https://oss-accelerate.aliyuncs.com
        :return:
        """
        # 生成签名URL时，OSS默认会对Object完整路径中的正斜线（/）进行转义，从而导致生成的签名URL无法直接使用。
        # 设置slash_safe为True，OSS不会对Object完整路径中的正斜线（/）进行转义，此时生成的签名URL可以直接使用。
        if accelerate:
            self.endpoint = accelreate_endpoint
            self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucketname, region=self.region)

        url = self.bucket.sign_url('GET', object_name, expires=expiration, slash_safe=True)
        # print('签名URL的地址为：', url)
        return url

    def get_objects_list(self,
                         num: int = 100):
        """
        列举指定数量的文件object
        :param num: 返回指定数量的文件object
        :return:
        """
        objects_list = []
        try:
            objects = list(islice(oss2.ObjectIterator(self.bucket), num))
            for obj in objects:
                self.logger.info(obj.key)
                objects_list.append(obj.key)
            return objects_list
        except oss2.exceptions.OssError as e:
            self.logger.error(f"Failed to list objects: {e}")

    def get_object_from_url(self,
                            file_url: str = None,
                            save_path: str = "d:/downloads/myfile.txt",
                            ):
        """
        使用requests库下载文件
        :param file_url:
        :param save_path:
        :return:
        """
        try:
            response = requests.get(file_url, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(4096):
                        f.write(chunk)
                self.logger.info("Download completed!")
            else:
                self.logger.info(f"No file to download. Server replied HTTP code: {response.status_code}")
        except Exception as e:
            self.logger.info("Error during download:", e)

    def get_object(self,
                   file_url:str = None,
                   from_file_path: str = None,
                   to_file_path: str = None,
                   resumable: bool = False,
                   breakpoints_info_path: str = '/temp',
                   multiget_threshold=100 * 1024,
                   part_size=100 * 1024,
                   progress_callback=percentage_callback,
                   num_threads=1
                   ):
        """
        断点续传下载
        :param file_url: 下载buket对象分享链接,(signed_url)
        :param from_file_path: yourObjectName填写Object完整路径，完整路径中不能包含Bucket名称，例如exampledir/exampleobject.txt
        :param to_file_path: yourLocalFile填写本地文件的完整路径，例如D:\\localpath\\examplefile.txt
        :param resumable: bool 断点续传下载开关
        :param breakpoints_info_path: 断点续传下载信息保存路径
        :param multiget_threshold: 指定当文件长度大于或等于可选参数multipart_threshold（默认值为10 MB）时，则使用断点续传下载。
        :param part_size: 设置分片大小，单位为字节，取值范围为100 KB~5 GB。默认值为100 KB
        :param progress_callback: 设置下载进度回调函数
        :param num_threads: 如果使用num_threads设置并发下载线程数，请将oss2.defaults.connection_pool_size设置为大于或等于并发下载线程数。默认并发下载线程数为1
        :return:
        """

        if file_url is not None:
            try:
                response = requests.get(file_url, stream=True)
                if response.status_code == 200:
                    with open(to_file_path, 'wb') as f:
                        for chunk in response.iter_content(4096):
                            f.write(chunk)
                    self.logger.info("Download completed!")
                else:
                    self.logger.info(f"No file to download. Server replied HTTP code: {response.status_code}")
            except Exception as e:
                self.logger.info("Error during download:", e)

        if from_file_path is not None:
            if resumable:  # 断点续传
                # 如果使用store指定了目录，则断点信息将保存在指定目录中。如未使用参数store指定目录，则会在HOME目录下建立.py-oss-upload目录来保存断点信息
                store = oss2.ResumableDownloadStore(root=breakpoints_info_path)
                # 如果使用num_threads设置并发下载线程数，请将oss2.defaults.connection_pool_size设置为大于或等于并发下载线程数。默认并发下载线程数为1。
                oss2.defaults.connection_pool_size = num_threads
                # yourLocalFile填写本地文件的完整路径，例如D:\\localpath\\examplefile.txt。如果未指定本地路径，则默认从示例程序所属项目对应本地路径中上传文件。
                oss2.resumable_download(self.bucket, key=from_file_path, filename=to_file_path,
                                        store=store,
                                        # 指定当文件长度大于或等于可选参数multipart_threshold（默认值为10 MB）时，则使用断点续传下载。
                                        multiget_threshold=multiget_threshold,
                                        # 设置分片大小，单位为字节，取值范围为100 KB~5 GB。默认值为100 KB。
                                        part_size=part_size,
                                        # 设置下载回调进度函数。
                                        progress_callback=progress_callback,
                                        # 如果使用num_threads设置并发下载线程数，请将oss2.defaults.connection_pool_size设置为大于或等于并发下载线程数。默认并发下载线程数为1。
                                        num_threads=num_threads)
            else:
                # 使用get_object_to_file方法将buket空间上object下载至本地文件
                self.bucket.get_object_to_file(key=to_file_path, filename=from_file_path)
            self.logger.info("Download completed!")

    def put_object(self,
                   upload_data: bytes | str = None,
                   from_file_path: str = None,
                   to_file_path: str = None,
                   resumable: bool = False,
                   breakpoints_info_path: str = '/temp',
                   multipart_threshold=100 * 1024,
                   part_size=100 * 1024,
                   progress_callback=percentage_callback,
                   num_threads=1
                   ):
        """
        上传文件到OSS,支持断点续传
        :param upload_data: 可以是byte,或者字符串的 data,直接上传,而不是文件上传
        :param from_file_path: 上传本地文件路径
        :param to_file_path: yourObjectName填写Object完整路径，完整路径中不能包含Bucket名称，例如exampledir/exampleobject.txt。
        :param resumable: 是否使用断点续传;
        :param breakpoints_info_path: 断点信息存储路径,如果指定了目录，则断点信息将保存在指定目录中。如未指定目录，则会在HOME目录下建立.py-oss-upload目录来保存断点信息
        :param multipart_threshold: 指定当文件长度大于或等于可选参数multipart_threshold（默认值为10 MB）时，则使用分片上传。
        :param part_size: 设置分片大小，单位为字节，取值范围为100 KB~5 GB。默认值为100 KB。
        :param progress_callback: 设置上传回调进度函数。
        :param num_threads: 如果使用num_threads设置并发上传线程数，请将oss2.defaults.connection_pool_size设置为大于或等于并发上传线程数。默认并发上传线程数为1。
        :return:
        """

        if upload_data is not None:
            # 填写Object完整路径和Bytes内容。Object完整路径中不能包含Bucket名称。
            self.bucket.put_object(key=to_file_path, data=upload_data)

        if from_file_path is not None:
            if resumable:  # 断点续传
                # 如果使用store指定了目录，则断点信息将保存在指定目录中。如未使用参数store指定目录，则会在HOME目录下建立.py-oss-upload目录来保存断点信息
                store = oss2.ResumableStore(root=breakpoints_info_path)
                # 如果使用num_threads设置并发上传线程数，请将oss2.defaults.connection_pool_size设置为大于或等于并发上传线程数。默认并发上传线程数为1。
                oss2.defaults.connection_pool_size = num_threads
                # yourLocalFile填写本地文件的完整路径，例如D:\\localpath\\examplefile.txt。如果未指定本地路径，则默认从示例程序所属项目对应本地路径中上传文件。
                oss2.resumable_upload(self.bucket, key=to_file_path, filename=from_file_path,
                                      store=store,
                                      # 指定当文件长度大于或等于可选参数multipart_threshold（默认值为10 MB）时，则使用分片上传。
                                      multipart_threshold=multipart_threshold,
                                      # 设置分片大小，单位为字节，取值范围为100 KB~5 GB。默认值为100 KB。
                                      part_size=part_size,
                                      # 设置上传回调进度函数。
                                      progress_callback=progress_callback,
                                      # 如果使用num_threads设置并发上传线程数，请将oss2.defaults.connection_pool_size设置为大于或等于并发上传线程数。默认并发上传线程数为1。
                                      num_threads=num_threads)
            else:
                # 使用put_object_from_file方法将本地文件上传至OSS
                self.bucket.put_object_from_file(key=to_file_path, filename=from_file_path)
            self.logger.info("upload completed!")



if __name__ == '__main__':
    endpoint = "https://oss-cn-hangzhou.aliyuncs.com"
    region = "cn-hangzhou"
    BucketName = "dt-drive"
    # # 从环境变量中获取访问凭证。运行本代码示例之前，请确保已设置环境变量OSS_ACCESS_KEY_ID和OSS_ACCESS_KEY_SECRET。
    # auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    # # yourBucketName填写存储空间名称。
    # bucket = oss2.Bucket(auth, endpoint, BucketName, region=region)

    bucket = BucketObject(endpoint, region, BucketName)
    objects_list = bucket.get_objects_list()
    url = bucket.get_signed_url(object_name=objects_list[0], expiration=7*24*3600, accelerate=False)
    print(url)
    upload_file_path = r"L:/Python_WorkSpace/modelscope/OSS_aliyun_client.py"
    # bucket.put_object(from_file_path=upload_file_path, to_file_path="OSS_aliyun_client.py")
    objects_list = bucket.get_objects_list()
    download_file_path = r"L:/temp/OSS_aliyun_client.py"
    bucket.get_object(from_file_path=objects_list[0], to_file_path=download_file_path,)
