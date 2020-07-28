import urllib2
import urllib
from bs4 import BeautifulSoup
import csv
import time
import random

baseurl='https://sg.linkedin.com/job/data-analyst-jobs-singapore/'
path='/Users/chaoranliu/Desktop/data.csv'

def fetchJobs(baseurl,path,nPages=3):
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

def addJD(data,path):
	links=[i['link'] for i in data]
	for i,url in enumerate(links):
		setDelay(5)
		response=valid(getResponse,url)
		data[i]['JD']=valid(getJD,response)
		print str(i+1)+'/'+str(len(links))+' pages completed.'
	print 'Comoplete!'
	write_csv(data,path)
	print 'Writing data to '+path+' is done!'
	return data


def getResponse(url):
	user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:39.0) Gecko/20100101 Firefox/39.0'
	headers={'User-Agent':user_agent}

	request=urllib2.Request(url,headers=headers)
	response=urllib2.urlopen(request).read()
	response=BeautifulSoup(response)
	return response

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


def valid(fn,x):
	try:
		return fn(x)
	except:
		return None

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

def getNextPage(page):
	try:
		return [p['href'] for p in page if 'next' in p['href']][0]
	except:
		return None

def setDelay(n):
	print 'wait ~'+n+' seconds ...'
	delay=(0.8+0.4*random.random())*n
	time.sleep(delay)

def write_csv(data,path):
	fieldnames=data[0].keys()
	with open(path,'w') as csvfile:
		writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
		writer.writeheader()
		for i in data:
			writer.writerow(i)

def getJD(response):
	return response.findAll('div',{'itemprop':'description'})[0].text.encode('utf-8')
