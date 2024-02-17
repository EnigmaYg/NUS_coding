import urllib.request
import ssl
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import re
from selenium.webdriver.chrome.options import Options
import time
import json
import os
import urllib.request
import ssl
import hashlib

def get_image_url(num, key_word):
    # box = driver.find_element_by_xpath('/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/textarea')
    box = driver.find_element(By.XPATH, '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/textarea')
    box.send_keys(key_word)
    # print(type(driver))
    # print(type(box))
    # driver.find_element_by_xpath('/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button').click()
    driver.find_element(By.XPATH,'/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button').click()
    time.sleep(3)
    # element = driver.find_element_by_tag_name("body")
    # img = driver.find_element_by_xpath('/body/div/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/span/div/div/div[1]/a')
    imgurl = []
    cnt = 0
    i = 1
    flag = 0
    while cnt != num:
        try:
            XPATH = '//*[@class="islrc"]/div[' + str(i) + ']/a'
            # driver.find_element_by_xpath(XPATH).click()
            driver.find_element(By.XPATH,XPATH).click()
        except:
            break
        time.sleep(5)
        try:
            # imgurl.append(driver.find_element_by_xpath('//*[@class="r48jcc pT0Scc iPVvYb"]').get_attribute('src'))
            imgurl.append(driver.find_element(By.XPATH,'//*[@class="sFlh5c pT0Scc iPVvYb"]').get_attribute('src'))
            time.sleep(1)
            imgurl.append(driver.find_element(By.XPATH, '//*[@class="Hnk30e indIKd"]').get_attribute('href'))
            i += 1
            cnt += 1
            driver.back()
            time.sleep(4)
        except:
            i += 1
            driver.back()
            time.sleep(4)
            pass
    driver.back()
    return imgurl


def get_htm(urls):
    t = len(urls) // 2
    for i in range(0, t):
        ssl._create_default_https_context = ssl._create_unverified_context
        url = urls[i*2 + 1]
        # url = r'https://www.pbs.org/newshour/world/israel-ramps-up-demolition-of-palestinian-homes-in-jerusalem'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'}
        req = urllib.request.Request(url=url, headers=headers)
        res = urllib.request.urlopen(req)
        html = res.read().decode('utf-8')
        match = re.search(r'(https?://\S+\.(jpg|jpeg|png))', urls[i*2])
        with open("page" + str(i) + ".html", 'w', encoding='utf-8') as f:
            f.write(html)
        if match:
            result = match.group(1)
            if result in html:
                print("T")
            else:
                print("F")
        else:
            if urls[i * 2] in html:
                print("T")
            else:
                print("F")


if __name__ == '__main__':
    ch_op = Options()
    # optional : do not want to show the ui
    # ch_op.add_argument('--headless')
    ch_op.add_argument('--disable-gpu')
    ch_op.add_argument('--no-sandbox')
    ch_op.add_argument('--disable-dev-shm-usage')
    ch_op.add_argument('--window-size=1200,1200')

    url = "https://images.google.com/"
    service = Service(executable_path=r'/usr/local/bin/chromedriver')
    driver = webdriver.Chrome(service=service, options=ch_op)
    # driver = webdriver.Chrome(service=service)
    # driver = webdriver.Chrome('D:\GeckoDriver\chromedriver')
    driver.get(url)
    # image_urls = get_image_url(3, "Israel bears full responsibility for drone attack on Lebanon, Hariri tells UN chief")

    tit = 'Israel demolishes Palestinians homes to expand settlements'
    amount = 100
    image_urls = get_image_url(amount, tit)
    image_urls_fin = []
    print(len(image_urls))
    amount = len(image_urls)//2
    for i in range(0, amount):
        image_urls_fin.append(image_urls[i*2])
        try:
            image_urls_fin.append(image_urls[i*2 + 3])
        except:
            pass
    image_urls_fin.pop(2*amount - 2)
    html = get_htm(image_urls_fin)
    with open('1.json', 'w') as f:
        json.dump(image_urls_fin, f, indent=2)
    driver.quit()
