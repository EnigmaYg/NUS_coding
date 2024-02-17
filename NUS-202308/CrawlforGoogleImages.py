from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import json
import os
import urllib.request
import uuid

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
    while cnt != 3:
        try:
            XPATH = '//*[@class="islrc"]/div[' + str(i) + ']/a'
            driver.find_element_by_xpath(XPATH).click()
            time.sleep(5)
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
    print(imgurl)
    driver.back()
    return imgurl

if __name__ == '__main__':
    ch_op = Options()
    # optional : do not want to show the ui
    ch_op.add_argument('--headless')
    ch_op.add_argument('--disable-gpu')

    url = "https://images.google.com/"
    driver = webdriver.Chrome('D:\GeckoDriver\chromedriver', options=ch_op)
    # driver = webdriver.Chrome('D:\GeckoDriver\chromedriver')
    driver.get(url)
    # image_urls = get_image_url(3, "Israel bears full responsibility for drone attack on Lebanon, Hariri tells UN chief")

    write_in = dict()
    count = 0
    time_start = time.time()
    with open('docs.json', 'r') as f:
        docs_data = json.load(f)
        for key, value in docs_data.items():
            if count == 20:
                break
            tit = value['Title']
            # tit.replace(" ", "+")
            # tit.replace("'", "%27")
            num = 3
            image_urls = get_image_url(num, tit)
            write_in[key] = image_urls
            count += 1
    time_end = time.time()
    print('time cost:', time_end - time_start, 's')
    json_str = json.dumps(write_in).replace("'", "\"")
    json_in = json.loads(json_str)
    with open('photo-url.json', 'w') as f:
        json.dump(json_in, f, indent=2)
    driver.quit()
