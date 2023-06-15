import os
import requests
import subprocess
import time
import argparse
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
    retries_url = 0
    while retries_url < max_retries:
        time.sleep(1)
        try:
            response = requests.get(url)
            break
        except requests.exceptions.RequestException:
            print('获取根路径失败:', url)
            retries_url += 1
            print(f'正在重试 ({retries_url}/{max_retries})...')
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
def batch_download_models(model_size):
    # 设置模型的文件夹
    download_folder = 'models/'+model_size+'-model'
    create_folder_if_not_exists(download_folder)
    # 设置调用列表
    model_list_large = {
        'resnet-50-224':'microsoft/resnet-50',
        'beit-large-224':'microsoft/beit-large-patch16-224-pt22k-ft22k',
        'convnext-large-224': 'facebook/convnext-large-224',
        'vit-large-384':'google/vit-large-patch32-384',
        'swin-large-384':'microsoft/swin-large-patch4-window12-384-in22k'
    }
    model_list_base = {
        'resnet-18-224':'microsoft/resnet-18',
        'beit-base-224':'microsoft/beit-base-patch16-224-pt22k-ft22k',
        'convnext-base-224':'facebook/convnext-base-224',
        'vit-base-224':'google/vit-base-patch16-224',
        'swin-base-384':'microsoft/swin-base-patch4-window12-384'
    }
    model_list_mobile = {
        'mobilevit-apple-288':'apple/mobilevit-small',
        'mobilenetv2-google-224':'google/mobilenet_v2_1.0_224',
        'mobilenetv3-timm-224':'timm/mobilenetv3_large_100.ra_in1k',
        'mobilenetv1-google-192':'google/mobilenet_v1_0.75_192',
        'mobilenet_s-timm-256':'timm/mobilevit_s.cvnets_in1k'
    }

    if model_size == 'large':
        for key in model_list_large:
            url = 'https://huggingface.co/' + model_list_large[key] + '/tree/main'
            download_pretrained_models(key, url, download_folder)
        print('Downloading done...')
    elif model_size == 'base':
        for key in model_list_base:
            url = 'https://huggingface.co/' + model_list_base[key] + '/tree/main'
            download_pretrained_models(key, url, download_folder)
        print('Downloading done...')
    elif model_size == 'mobile':
        for key in model_list_mobile:
            url = 'https://huggingface.co/' + model_list_mobile[key] + '/tree/main'
            download_pretrained_models(key, url, download_folder)
        print('Downloading done...')
    else:
        print('Please choose large/base/mobile one of them...')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Translate articles using a pretrained model.")
    parser.add_argument("--models_size", type=str, help="The size of models, you can choose large/base/mobile.", required=True)
    args = parser.parse_args()
    batch_download_models(args.models_size)
    #batch_download_models("mobile")
    # save_dict_to_txt(get_model_list(), 'modelList.txt')
    # print('done...')
