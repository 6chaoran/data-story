# Job Hunting Like A Data Analyst
## Motivation:
I'm currently suffering a tough time in looking for a data analyst job.  
Instead of doing it in a traditional way, I am thinking why not do the job hunting just like a data analyst, by making use of the advantages of data science. 

Always, the first step to flight a battle is to know your enemy. As I'm looking for a job in Singapore/China, the first thing I would like to explore is the job market in these areas.   
I'm interested to know about:

* Who is hiring data analyst
* Which cities need data analyst most
* What skills are expected
* How is a data analyst described

What I will do is to start with LinkedIn Job Search to find the job posting about Data Analyst, using data science of course!

## Information Collection:
### What You Need:
 * Python 2.x
 * Chrome/Firefox Browser
 * package urllib2
 * package bs4
	
### Essentials of a simple Web Scraping
#### 1. Fetching URLs
The basic idea is to get URL for the first page and download the content of page. Then get the URL of next page and repeat.
#### 2. Mimic to Browsers
Sometimes the web server won't recognise the python browser and hence it is always a good practice to mimic as another popular web browser when doing the web scraping.  `urllib2` has a 'headers' parameter, which can be used to define the browser.
#### 3. Setting a Delay
Most websites don't like web spiders and will prevent too frequent request from the same IP address.  So it is better to set a time delay between each request.
#### 4. Parsing HTML
After downloading the html of the page, we are not going to store it directly. Instead we will extract the useful information (e.g. text, links), store it and throw away the page html.
#### 5. Saving into Data File
After the information is extracted, we output the data into csv/txt file for easier future use.

### Scraping LinkedIn Job Search
To find the job post on LinkedIn is simply go to LinkedIn Job Search site and input keywords data analyst and Singapore, and then website will direct us to a new URL:
"https://sg.linkedin.com/job/data-analyst-jobs-singapore/", which is more or less like the screenshot below.
![image](https://6chaoran.files.wordpress.com/2015/08/linkedinjobsearch.jpg)
Instead of browsing every page to look for the ideal job post, I would like to use python to browse the pages for us. 

#### Fetch the first page
Using python urllib2 package, we can simply write a function to send HTTP request, download the page, read the content and parse the html to return the response.

	import urllib2
	from bs4 import BeautifulSoup
	def getResponse(url):
		user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:39.0) Gecko/20100101 Firefox/39.0'
		headers={'User-Agent':user_agent}
		# define the request
		request=urllib2.Request(url,headers=headers)
		# request, download and read the content
		response=urllib2.urlopen(request).read()
		# parse the html using BeautifulSoup
		response=BeautifulSoup(response)
		return response

It is import to include a header of user-agent in the request, otherwise you will get an error saying request is denied. To find the header information, you can go to Firefox->Tools->Web Developer->Network and it is under the header tab.
![image](https://6chaoran.files.wordpress.com/2015/08/browserheader.jpg)

#### Extract the useful information
Now we already have the html file ready and we can use BeautifulSoup findAll/findNext function to filter by the html tag. 
For example, when we need to get the Job Title from the page, we can use Firefox inspector (Firefox->Tools->Web Developer->Inspector) to easily locate the element.
![image](https://6chaoran.files.wordpress.com/2015/08/htmlselector.jpg)

By pointing to the job title, the h2 tag is highlighted.    
We can just call findNext/findAll('h2').text to extract the text in h2 tag. Similarly we can call findNext/findAll('h2')['href'] to extract the attribute, which is the link of the job post in this case.    
There are many possibilities of h2 tags other the job titles, so I would like to refine the selection to the job post main content by filtering ('ul',{'class':'jobs'}), which means ul tag with attribute 'class' and value 'jobs'.     
Within the job post content, we do another selection by filtering ('ul',{'class':'jobs'}) and it will generate a list of html contains 25 elements. 

	bassurl = 'https://sg.linkedin.com/job/data-analyst-jobs-singapore/'
	## get the response html from the page
	response=getResponse(baseurl)
	## refine the selection of job posting
	content=response.findAll('ul',{'class':'jobs'})[0]
	## select job lists
	jobs=content.findAll('li',{'class':'job'})
In order to get the basic information of job title, link, company name, post date and company location, I defined 4 functions to loop over the `jobs` list.    
Using encode('utf-8') can prevent the encode error, when writing the data into csv file.

	def getTitle(x):
	return x.findNext('h2').text.encode('utf-8')

	def getLink(x):
		return x.findNext('h2').findNext('a')['href']

	def getCompany(x):
		return x.findNext('a',{'class':'company'}).text.encode('utf-8')

	def getDate(x):
		return x.findNext('span',{'itemprop':'datePosted'}).text.encode('utf-8')

	def getLocation(x):
		return x.findNext('span',{'itemprop':'addressLocality'}).text.encode('utf-8')

The last and important information we need to get is the URL of next page, so that we can automatic the web scraping.
try/except in python means to run the try clause  and run the except clause only if error occurs.  This is added in case the loop comes to the last page.

	def getNextPage(page):
	try:
		return [p['href'] for p in page if 'next' in p['href']][0]
	except:
		return None
	
#### Store the data
There are some types of collection in python, e.g. list, set, dictionary, tuple, among which we are going to use combination of list and dictionary to store the data. Because it is more convenient to write to csv file using DictWriter function or easier to convert to pandas DataFrame type.   
The format of python dictionary is {key1:value1,key2:value2,...}. Each record of the data will be a dictionary with the column names as the key and information as the value. 
Then each dictionary will be appended as a list to form a collection.   
Here an additional valid function is introduced, because some information may be missing for certain record and it will return None if such case happens.

	data=[]
	for i in jobs:
		row={}
		row['title']=valid(getTitle,i)
		row['company']=valid(getCompany,i)
		row['date']=valid(getDate,i)
		row['location']=valid(getLocation,i)
		row['link']=valid(getLink,i)
		data.append(row)
		
	def valid(fn,x):
		try:
			return fn(x)
		except:
			return None
			
#### Put them together
We now have all the pieces ready and we can write a function called `fetchPage` to group them together.

	def fetchPage(baseurl):
		response=getResponse(baseurl)
		## page body of job posting
		content=response.findAll('ul',{'class':'jobs'})[0]
		## page navigation bar
		page=response.findAll('div',{'class':'pagination'})[0].findAll('a',{'rel':'nofollow'})
		## job lists
		jobs=content.findAll('li',{'class':'job'})
		## get url for next page
		nextPageUrl=getNextPage(page)
		## store information into list data
		data=[]
		for i in jobs:
			row={}
			row['title']=valid(getTitle,i)
			row['company']=valid(getCompany,i)
			row['date']=valid(getDate,i)
			row['location']=valid(getLocation,i)
			row['link']=valid(getLink,i)
			data.append(row)
	
		return data,nextPageUrl

#### Loop it over the pages
Since we already have the function to return the data and URL of nextPage, we can just write a `for` loop to get as many pages as we want.    
One thing to mention is that a time delay is recommended to set between the requests. In python, we can use `time.sleep(x)` function from time module, which means pause the program for x seconds. A constant time interval may look more like a robot, so I set the delay time to be a random variable. It can be accomplished by `random.random()` function, which generates a [0,1) float number.

	import time
	import random
	
	def setDelay(n):
		print 'wait ~'+n+' seconds ...'
		## I set +/- 20% range of randomness
		delay=(0.8+0.4*random.random())*n
		time.sleep(delay)
  
 Loop the `fetchPage' function, we will have the code like:
 
	def fetchPages(baseurl,path,nPages=3):
		print 'Start:'
		data0,nextPageUrl=fetchPage(baseurl)
		print 'Page 1 completed'
		if nextPageUrl:
			pass
		else:
			return 'nextUrl is missing in first page!'
		for i in range(2,nPages+1):
			if nextPageUrl:
				setDelay(5)
				data,nextPageUrl=fetchPage(nextPageUrl)
				print 'Page '+str(i)+' completed'
				data0+=data
			else:
				print 'nextUrl is missing in page '+str(i)+'!'
				break
		write_csv(data0,path)
		print 'Writing data to '+path+' is done!'
		return data0
#### Write to csv
If you carefully exam the code in previous section, you will noticed the `write_csv` function is actually not yet defined.
Here is the code, which uses `csv.DictWriter` to write into csv file row by row.

	def write_csv(data,path):
		fieldnames=data[0].keys()
		with open(path,'w') as csvfile:
			writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
			writer.writeheader()
			for i in data:
				writer.writerow(i)


Browsing hundreds of job post is just as simple as one sentence:

	data=fetchPages(url, '.../data.csv',nPages=20)
	
Now you already got the 500 job posts in 2 mins!