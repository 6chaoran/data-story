---
title: "Web Scraping of JavaScript website"
date: 2019-08-25 19:21:00 +0800
toc: true
toc_sticky: true
categories:
  - web-scraping
tags:
  - python
---

In this post, I’m using `selenium` to demonstrate how to web scrape a JavaScript enabled page.

If you had some experience of using python for web scraping, you probably already heard of `beautifulsoup` and `urllib`. By using the following code, we will be able to see the HTML and then use HTML tags to extract the desired elements. However, if the web page embedded with JavaScript, you will notice that some of the HTML elements can’t be seen from beautiful soup, because they are render by the JavaScript. Instead you will only see the `<script>` tags, which indicate the JavaScript codes are placed.

<!--more-->

```python
urlpage = 'https://apps.polkcountyiowa.gov/PolkCountyInmates/CurrentInmates/' 
# download the web page
f = urllib.request.urlopen(urlpage) 
# extract the html text
html = f.read()
# parse html using beautifulsoup
bs = BeautifulSoup(html)
```
the desired html elements are rendered from the `<script>`, so an alternative is need to this page.
![](https://raw.githubusercontent.com/6chaoran/data-story/master/tutorial/selenium_web_scrape/image/3_js_script.png)


## Procedures of Web-Scraping using Selenium

### 1. Prerequisite

* download the chrome driver from [here](https://chromedriver.chromium.org/downloads)
* current stable version is 76.0.3809.126
* choose your Operating System (mac/windows/linux)
* extract the webdriver to `CHOME_DRIVER` (e.g. `./chromedriver`)

### 2. Launch the Chrome Driver

use `selenium` to launch the a chrome browser, by calling `webdriver.Chrome()`.
A blank chrome window should pop up.

```python
from selenium import webdriver
CHROME_DRIVER = './chromedriver'
# run chrome webdriver from executable path of your choice
driver = webdriver.Chrome(executable_path = CHROME_DRIVER)
```
Now, let's load the page we want to extract.

```python
# load web page
urlpage = 'https://apps.polkcountyiowa.gov/PolkCountyInmates/CurrentInmates/' 
driver.get(urlpage)
print('waiting 15s for page loading')
# wait for 15 seconds to allow the page completely loaded
time.sleep(15)
```

use `driver.quit()` to close the browser when you are done with testing.


### 3. Parse the Webpage

`selenium` provides multiple ways to locate the elements of the HTML. By using Chrome `Developer Tools` (Chrome > More tools > Developer tools), we can easily locate the HTML elements. 
For example, we're going to extract the link of `Details`, so we point the HTML element and copy the Xpath location. 

![](https://raw.githubusercontent.com/6chaoran/data-story/master/tutorial/selenium_web_scrape/image/1_extract_link.png)

In `selenium`, we can call `find_elements_by_xpath` to extract all elements that matching the xpath pattern.

```python
xpath = '//*[@id="DataTables_Table_0"]/tbody/tr[1]/td[1]/a'
results = driver.find_elements_by_xpath(xpath)
# results is a list, because find_elements_by_xpath look for all items matching the xpath.
# if use find_element_by_xpath, it returns the first item matches the xpath.
len(results)
# 1
results[0].get_attribute('href') 
# https://apps.polkcountyiowa.gov/PolkCountyInmates/CurrentInmates/Details?Book_ID=299591
```

It's worth noticing that, the xpath pattern is too specific and only returns the first link instead of all the links. Therefore we need to generalize the xpath pattern, to capture all the links.

Let's trace back the upper levels of the xpath. Instead of using `tr[1]` to extract the first row, we use `*[contains(@role,'row')]` to capture all the rows contains the class `role='row'`.

![](https://raw.githubusercontent.com/6chaoran/data-story/master/tutorial/selenium_web_scrape/image/2_extract_row.png)

Then in each element, we use `td/a` xpath to locate the `<a> tags`. Because the number of links is relative big, a `tqdm` progress bar is also added to show the progress of extraction.

```python
xpath = "//*[@id='DataTables_Table_0']/tbody//*[contains(@role,'row')]"
results = driver.find_elements_by_xpath(xpath)
len(results)
# 938
results[0].find_element_by_xpath('td/a').get_attribute('href')
# https://apps.polkcountyiowa.gov/PolkCountyInmates/CurrentInmates/Details?Book_ID=299591

# add progress bar to extract all links
from tqdm import tqdm
links = []
for result in tqdm(results):
	links.append(result.find_element_by_xpath('td/a').get_attribute('href'))
```

### 4. save the data

Finally, we can save the links to a csv for later usage.

```python
import pandas as pd
df_links = pd.DataFrame({'links':links})
# save data to csv
df_links.to_csv('./links.csv', index = False)
# close the browser
driver.quit()
```
