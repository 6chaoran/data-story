---
title: "Job Hunting Like A Data Analyst (Part II)"
date: 2015-09-07 16:16:01 +0800
categories: 
  - exploratory analysis
tags: 
  - python
toc: true
toc_sticky: true
---

Continued with [previous post](https://6chaoran.github.io/DataStory/job-hunting-web-scraping/), I’ve added some additional lines of codes to fetch the job description of each job post. This will take a bit longer time, which is about (1.5 hour) for me, because I set a delay of ~10 seconds between each request.   
This week I will continue with overview picture of the job market of Data Analyst and develop a simple recommender based on skill and experience requirement.

## 0. Tools

1. python 2.7
2. python package: pandas
3. python package: re

## 1. Job Market Overview

### data preparation:

After some time of web scraping, we will have a quite clean dataset, which consists of a big chunk of text job description. What I got is something like below:   

![image](https://6chaoran.files.wordpress.com/2015/09/job-description1.jpg?w=700)   

Loading data is very simple in pandas compared with package csv.

```python
import pandas as pd
csv_file='/Users/6chaoran/Desktop/sgJobs.csv'
 
# pandas csv loading
# similar with read_csv in R readr package
data=pd.read_csv(csv_file)
```
### company overview

Let’s create a quick function to plot a bar chart showing the frequency of the companies that are hiring Data Analyst.

```python
def freqPlot(df,col,title,n=20):
    # value_counts in pandas is simlar with table() in R
    # count each element in the Series
    freqList=df[col].value_counts()
 
    # I want to see the top 20 or less category
    n=min(len(freqList),n)
    freqList=freqList[:n]
    freqList.sort()
 
    # plot the horizontal barplot directly from the pandas DataFrame/Series
    return freqList.plot(kind='barh',title=title)
```

Now we can just type

```python
freqPlot(data,'company','company overview for Data Analyst job market')
```

to plot the chart like this:   

![image](https://6chaoran.files.wordpress.com/2015/09/company_barplot.png?w=700)   

Not surprisingly, mostly big IT companies and banks are hiring Data Analysts.

### job title overview

Since we have already created the barplot function, we can make use of it to explore the other columns in the data frame.   

![image](https://6chaoran.files.wordpress.com/2015/09/job_title.png?w=700)

The most common terms referred to a Data Analyst could be Business Analyst or Data Analyst, which are similar but actually difference in terms of job scope.   
The Job title is very industry specialized. The categories could be further cleaned, but it requires some text processing tools and I don’t see much value from doing that.

### experience requirement overview

There is a big chunk of text (Job Description) that we haven’t touched. So let’s get some useful information from that.   
The experience requirement is usually statement in a sentence like “requires at least xx years of experience in xxx industry”. So my idea is catch the patter ‘xx years’ using regular expression, which can be used with re package in python.   
some side notes of regular expression:   
In regular expression,   

* `.` means any single character, so I will use ‘….years’ pattern to catch either 1-9 years or 10+ years.
* `[0-9]` means any number (from 0-9)
* `^` inside `[]` means negation

```python
# load the regular expression package
import re
def getExperience(text):
    # find all the '...years' patterns in the job description
    years=re.findall('....years',text)
    def yearToNumber(years):
        try:
            #remove non-numeric character and then convert to integer 
            return int(re.sub('[^0-9]','',years))
        except:
            return None
    years=map(yearToNumber,years)
    if len(years)>0:
        return max(years)
    else:
        return None
 
# more than 20 years experience as the requirement looks unrealistic    
# set to NA for the cases
data.loc[data.experience>20,'experience']=None
 
# plot the bar chart
freqPlot(data,'experience','experience requirement for Data Analyst')
```

![image](https://6chaoran.files.wordpress.com/2015/09/experience.png?w=700)

### skill requirement overview

To get the skills from the job description, we are going to something a bit more complicated.

#### 1. tokenize the text

The principal idea to split the sentence by space, punctuation or other special character to get a bag of words, from which we can count the frequency, analyse the sentiment and some more.   

```python
import re
def cleanText(text):
    # convert all characters to lowercase
    text=text.lower()
 
    # keep only numbers and alphabets, 
    # replace the others with space
    text=re.sub('\W',' ',text)
 
    # split the text with space
    words=text.split(' ')
 
    # return a list of the unique words
    return list(set(words))
```

#### 2. vectorize the feature

As I’ve been researching on data science for quite a while, I already have a list of skills that is frequently possessed by Data Analysts. Vectorization means to convert the skills requirement in each data record into the binary vector of the pre-defined skills. Let’s say the skills are defined as [‘excel’,’r’,’sql’,’python’]. If the job post only requires excel and sql, then it will be converted into vector[1,0,1,0].    
Let’s put it into code:

```python
def convertTextFeature(text,feature):
    featureVector={}
    for f in feature:
        if f in text:
            featureVector[f]=1
        else:
            featureVector[f]=0
    return featureVector
```

#### 3. aggregation

In order to find the popularity of the skills that are needed. We can simply find the summation of each skill.

```python
# pre-defined skills 
skills=['excel','r','sql','python','tableau','d3','qlikview','hadoop','matlab','scala','sas','spss']
 
# convert JD to a bag of words
words=data['JD'].map(cleanText)
 
# vectorize the skills in data record
featureSkill=pd.DataFrame([convertTextFeature(text,skills) for text in words])
 
# sum the skill across the row
skillList=featureSkill.sum(axis=0)
skillList.sort()
skillList.plot(kind='barh',title='the skill requirement for a data analyst')
```

The skills popularity is shown below:   

![image](https://6chaoran.files.wordpress.com/2015/09/skills.png?w=700)   

Though there are a lot of advanced analytical software available in the market, Excel is still the most widely used tool. And after seeing this, I decide to pick up the SQL skills.   
   
The data set can be found [here](https://raw.githubusercontent.com/6chaoran/DataStory/master/JobHuntingLikeADataAnalyst/sgJobs.csv).   
The python code can be found [here](https://github.com/6chaoran/DataStory/blob/master/JobHuntingLikeADataAnalyst/JobMarket.py).   





