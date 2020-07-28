import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from datetime import datetime
import os
import pytz

utc =pytz.utc
sgp = pytz.timezone('Singapore')
today = utc.localize(datetime.today())
today_in_sgp = today.astimezone(sgp)
suffix = today_in_sgp.strftime('%Y%m%d')
print('today is {}'.format(suffix))

URL = 'https://voice.baidu.com/act/newpneumonia/newpneumonia/'

driver = webdriver.Chrome('./chromedriver.exe')
driver.get(URL)


all_states = driver.find_elements_by_xpath(
'//*[@id="nationTable"]/table/tbody/tr')

# close 1st state
all_states[0].click()


# record state level data
all_states = driver.find_elements_by_xpath(
'//*[@id="nationTable"]/table/tbody/tr')
states = [i.text for i in all_states]


# click the state tag to expand the city view
for state in all_states:
    state.click()

# record city level data
cities = []
all_cities = driver.find_elements_by_xpath(
'//*[@id="nationTable"]/table/tbody/tr/td/table')
for city in all_cities:
    rows = city.find_elements_by_xpath('./tbody/tr')
    value = [i.text for i in rows]
    cities +=value


expand_btn = driver.find_element_by_xpath(
    '//*[@id="foreignTable"]/div/span')
expand_btn.click()

# record country level data
countries = driver.find_elements_by_xpath(
'//*[@id="foreignTable"]/table/tbody/tr')
countries = [i.text for i in countries]

# close browser
driver.close()

print(cities[:3])
print(states[:3])
print(countries[:3])


# post process data
def process_raw(x):
    name,v = x.split('\n')
    confirm, recover, death = v.split(' ')
    return {'name':name, 
            'confirmed':confirm, 
            'recovered': recover, 
            'death': death}


cities = [process_raw(i) for i in cities]
df_cities = pd.DataFrame(cities)
df_cities.head()

states = [process_raw(i) for i in states]
df_states = pd.DataFrame(states)
df_states.head()

countries = [process_raw(i) for i in countries]
df_countries = pd.DataFrame(countries)
df_countries.head()

# save data
output_dir = 'covid19_data'
output = os.path.join(output_dir, suffix)

if not os.path.exists(output):
    os.makedirs(output, exist_ok = True)
    
df_cities.to_csv(
    '{}/cities.csv'.format(output), 
    index = False, encoding='utf-8')

df_states.to_csv(
    '{}/states.csv'.format(output), 
    index = False, encoding='utf-8')

df_countries.to_csv(
    '{}/countries.csv'.format(output), 
    index = False, encoding='utf-8')

# combine data
def read_each(path, part = 'cities.csv'):
    data = pd.read_csv(os.path.join(output_dir, path, part))
    data['added_date'] = path
    return data
    
def combine(output_dir):
    dirs = os.listdir(os.path.join(output_dir))
    data = {}
    for part in ['cities.csv','states.csv','countries.csv']:
        data[part] = []
        for d in dirs:
            each = read_each(d, part)
            data[part].append(each)
        data[part] = pd.concat(data[part])
    return data

print('combine data')
data = combine(output_dir)
for k,v in data.items():
    print("{}: {}".format(k, str(v.shape)))
    v.to_csv('summary_'+k, index = False)