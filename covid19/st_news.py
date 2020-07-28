# scrap COVID-19 news from strait times

from selenium import webdriver
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pytz
import time
import os
import re

CHROME_DRIVER = './chromedriver'
urlpage = 'https://www.straitstimes.com/global'
output_dir = './st_news'

def parse_new_i(x):
    title = x.get_attribute('title')
    link = x.get_attribute('href')
    return {'title': title, 'link': link}

def fetch_new_content(url):
    driver.get(url)
    res = driver.find_elements_by_xpath('//div[@itemprop="articleBody"]/p')
    return '\n'.join([i.text for i in res])

if __name__ == '__main__':

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    driver = webdriver.Chrome(executable_path = CHROME_DRIVER)
    driver.get(urlpage)
    time.sleep(5)
    news = driver.find_elements_by_xpath('//a[@class="block-link"]')

    results = []
    for i, x in enumerate(news):
        try:
            results.append(parse_new_i(x))
        except:
            continue

    # clean up the links
    res = pd.DataFrame(results)
    res = res.loc[res.title.map(lambda i: len(i) > 20), :]
    res = res.loc[~pd.DataFrame.duplicated(res),:]
    res = res.loc[res.link != 'javascript:void(0)',:]

    # fetech contents
    news_content = []
    for url in tqdm(res.link):
        try:
            row = {'link': url, 'content': fetch_new_content(url)}
            news_content.append(row)
        except:
            continue
    df_content = pd.DataFrame(news_content)

    # join together
    data = res.set_index('link').join(df_content.set_index('link'), how = 'inner')
    data = data.reset_index('link')

    # save data
    print('writing results csv')
    utc =pytz.utc
    sgp = pytz.timezone('Singapore')
    today = utc.localize(datetime.today()).astimezone(sgp)
    data.to_csv('{}/{}'.format(output_dir, today.strftime('%Y_%m_%d.csv')))

    # filter covid-19 news
    data = []
    for i in os.listdir(output_dir):
        add = pd.read_csv(os.path.join(output_dir, i), index_col = 0)
        add['date'] = re.sub('.csv','',i)
        data += [add]
    data = pd.concat(data)
    data = data[['title','link','date']].loc[data.title.str.contains('covid-19|coronavirus',case = False),:]
    data.sort_values('date', ascending=False).to_csv('./news_feed.csv')
    print("job's done")
