from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import json
import os
import urllib.request
import ssl
import hashlib
import uuid

def get_md5(url):
    if isinstance(url, str):
        url = url.encode("utf-8")
    m = hashlib.md5()
    m.update(url)
    return m.hexdigest()

def get_image_url(num, key_word):
    box = driver.find_element_by_xpath('/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/textarea')
    box.send_keys(key_word)
    # print(type(driver))
    # print(type(box))
    driver.find_element_by_xpath('/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button').click()
    time.sleep(3)
    element = driver.find_element_by_tag_name("body")
    # img = driver.find_element_by_xpath('/body/div/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/span/div/div/div[1]/a')
    imgurl = []
    cnt = 0
    i = 1
    while cnt != num:
        try:
            XPATH = '//*[@class="islrc"]/div[' + str(i) + ']/a'
            driver.find_element_by_xpath(XPATH).click()
        except:
            break
        time.sleep(5)
        try:
            imgurl.append(driver.find_element_by_xpath('//*[@class="r48jcc pT0Scc iPVvYb"]').get_attribute('src'))
            i += 1
            cnt += 1
            driver.back()
            time.sleep(4)
        except:
            i += 1
            driver.back()
            time.sleep(4)
            pass
    '''imgurl.append(driver.find_element_by_xpath('//*[@class="r48jcc pT0Scc iPVvYb"]').get_attribute('src'))
    driver.find_element_by_xpath('//*[@class="islrc"]/div[1]/a').click()
    time.sleep(5)
    imgurl = [driver.find_element_by_xpath('//*[@class="r48jcc pT0Scc iPVvYb"]').get_attribute('src')]
    # imgurl = [driver.find_element_by_xpath('//*[@class="MAtCL PUxBg"]/a/img').get_attribute('src')]
    # imgurl = [driver.find_element_by_xpath('//*[@id=islsp]/div[2]/div/div/div/div[1]/c-wiz/div/div/div/div[3]/div[1]/a/img').get_attribute('src')]
    driver.back()
    time.sleep(4)
    driver.find_element_by_xpath('//*[@class="islrc"]/div[2]/a').click()
    time.sleep(5)
    imgurl.append(driver.find_element_by_xpath('//*[@class="r48jcc pT0Scc iPVvYb"]').get_attribute('src'))
    driver.back()
    time.sleep(4)
    driver.find_element_by_xpath('//*[@class="islrc"]/div[3]/a').click()
    time.sleep(5)
    imgurl.append(driver.find_element_by_xpath('//*[@class="r48jcc pT0Scc iPVvYb"]').get_attribute('src'))'''
    # print(imgurl)
    driver.back()
    return imgurl

def images_downloads(**url):
    for key, value in url.items():
        l = []
        for i in value:
            '''ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE'''
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'}
                req = urllib.request.Request(url=i, headers=headers)
                res = urllib.request.urlopen(req).read()
                path = "/storage_fast/ccye/zmyang/images/" + key_
                with open(path + '.png', 'wb') as f:
                    f.write(res)
                    f.close()
            except:
                print(key+i)
                pass
            key_ = get_md5(i)
            l.append(key_)
            write_in_in[key_] = i
        write_in_[key] = l

write_in_ = dict()
write_in_in = dict()
if __name__ == '__main__':
    ch_op = Options()
    # optional : do not want to show the ui
    ch_op.add_argument('--headless')
    ch_op.add_argument('--disable-gpu')

    url = "https://images.google.com/"
    driver = webdriver.Chrome('/usr/bin/chromedriver', options=ch_op)
    # driver = webdriver.Chrome('/usr/bin/chromedriver')
    driver.get(url)
    # image_urls = get_image_url(3, "Israel bears full responsibility for drone attack on Lebanon, Hariri tells UN chief")

    write_in = dict()
    count = 0
    time_start = time.time()
    with open('/storage_fast/ccye/LoGo/data/EGIRIS_CE/docs.json', 'r') as f:
        docs_data = json.load(f)
        item = list(docs_data.items())
        for key, value in item[219000:]:
            if count == 2000:
                break
            print(count)
            tit = value['Title']
            # tit.replace(" ", "+")
            # tit.replace("'", "%27")
            num = 3
            image_urls = get_image_url(num, tit)
            write_in[key] = image_urls
            count += 1
    time_end = time.time()
    print('time cost:', time_end - time_start, 's')
    json_str = json.dumps(write_in)
    json_in = json.loads(json_str)
    # images_downloads(**json_in)
    with open('photo-url219000-220999.json', 'w') as f:
        json.dump(json_in, f, indent=2)
    # with open('MD5doc2pho0-199.json', 'w') as f:
    #    json.dump(write_in_, f, indent=2)
    # with open('MD5pho2pho0-199.json', 'w') as f:
    #    json.dump(write_in_in, f, indent=2)
    driver.quit()
