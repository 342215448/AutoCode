import os
import requests
import subprocess
import time
from bs4 import BeautifulSoup


# 创建文件夹的函数
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功")
    else:
        print(f"文件夹 '{folder_path}' 已经存在")

# 下载预训练好的模型到指定目录下 每次调用相当于下载了一个对应的模型
def download_pretrained_models(model_name, url, download_folder, max_retries=20):
    # download_folder = download_folder + '/' + url.split('/')[-3]
    download_folder = download_folder + '/' + model_name
    create_folder_if_not_exists(download_folder)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 查找具有特定标题的所有a标签
    a_tags = soup.find_all('a', title="Download file")

    for a_tag in a_tags:
        href = 'https://huggingface.co' + a_tag.get('href')
        retries = 0
        while retries < max_retries:
            time.sleep(3)
            try:
                # 使用subprocess模块调用wget命令，并设置重试次数
                subprocess.run(['wget', href, '-P', download_folder], check=True)
                print('下载成功:', href)
                break
            except subprocess.CalledProcessError:
                print('下载失败:', href)
                retries += 1
                print(f'正在重试 ({retries}/{max_retries})...')

        if retries == max_retries:
            print(f'无法下载文件: {href}')

# 批量下载所有指定的模型
def batch_download_models():
    # 设置模型的文件夹
    download_folder = 'models/large-model'
    create_folder_if_not_exists(download_folder)
    # 设置调用列表
    model_list = {
        'beit-large-224':'microsoft/beit-large-patch16-224-pt22k-ft22k'
        #'convnext-large-224': 'facebook/convnext-large-224',
        #'vit-large-384':'google/vit-large-patch32-384',
        #'swin-large-384':'microsoft/swin-large-patch4-window12-384-in22k'
    }

    for key in model_list:
        url = 'https://huggingface.co/' + model_list[key] + '/tree/main'
        download_pretrained_models(key, url, download_folder)
    print('Downloading done...')



if __name__ == '__main__':
    
    batch_download_models()
    # save_dict_to_txt(get_model_list(), 'modelList.txt')
    # print('done...')
