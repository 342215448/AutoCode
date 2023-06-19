import re
import requests
import os
from PIL import Image

def downloadPic(url,file,keyword,num):
    try:
        if url is not None:
            pic = requests.get(url, timeout=7)
        else:
            return False
    except BaseException:
        print('错误，当前图片无法下载')
        return False
    else:
        try:
            # 定义下载到的图片的命名格式
            string = file + r'\\' + keyword + '_' + str(num+1) + '.jpg'
            # 以二进制写的方式打开该路径
            fp = open(string, 'wb')
            # 将下载到的文件内容写入到该路径下
            fp.write(pic.content)
            fp.close()
            # 读取该路径的图片
            img = Image.open(string)
            # 如果该图片的宽度小于30，就删除该图片
            if img.width < 30:
                img.close()
                os.remove(string)
                return False
            img.close()
        except BaseException:
            print('错误，当前图片无法下载')
            return False
    return True

def downloadPictures(keyword,numPicture,url,A,file):
    num = 0 # 记录图片下载数量
    t = 0
    print('找到关键词:' + keyword + '的图片，即将开始下载图片...')
    while num < numPicture:
        Url = url + str(t)
        try:
            Request = A.get(Url, timeout=7, allow_redirects=False)
        except BaseException:
            t = t + 60
            continue
        else:
            # 获取网页源码
            result = Request.text
            # 正则表达式取出图片的url
            pic_url = re.findall('"objURL":"(.*?)",', result, re.S)
            if len(pic_url) == 0:
                break
            else:
                # 将url存入list
                for each in pic_url:
                    print('正在下载第' + str(num + 1) + '张图片，图片地址:' + str(each))
                    if(downloadPic(each,file,keyword,num)):
                        num += 1
                    if num >= numPicture:
                        break
                # 翻页
                t = t + 60

def crawPictures(words_list,numPictureList):
    '''
    Introduction:
        爬取百度图片并下载到images文件夹下
    Args:
        words_list:关键词列表
        numPictureList:每个关键词需要下载的图片数量
    '''
    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
        'Upgrade-Insecure-Requests': '1'
    }

    # requests.Session():请求会话，提供 cookie 持久性、连接池和配置
    A = requests.Session()

    A.headers = headers
    ###############################
    for word,numPicture in zip(words_list,numPictureList):
        url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&pn='

        file = os.path.abspath(os.path.dirname(os.getcwd())) + '/images/' + word
        # 判断该文件是否已存在
        while os.path.exists(file):
            print('该文件已存在，请重新输入')
            file = input('请建立一个存储图片的文件夹，)输入文件夹名称即可')
        os.mkdir(file)
        # 下载图片
        downloadPictures(word,numPicture,url,A,file)
        print("%s类图片下载完成" % word)

    print('搜索结束，感谢使用')

if __name__ == '__main__':
    word_list = ['性感的女生','可爱的女生']
    numPicture_list = [100,200]
    crawPictures(word_list,numPicture_list)
