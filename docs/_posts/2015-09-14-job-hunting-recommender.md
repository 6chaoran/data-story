---
title: "Job Hunting Like A Data Analyst (Part III)"
date: 2015-09-14 16:16:01 +0800
categories: 
  - recsys
tags:
  - python
toc: true
toc_sticky: true
---

Continued with previous post – Explore the Job Market, this week I am going to develop a simple recommender system to find a suitable job .

## Recommender

Let’s talk some background of recommendation system. A typical example of recommendation could be product recommended in the sidebar at Amazon or people you may know in Facebook.   
Usually we can categorised recommender into two types:

### 1. Content Based Recommendation:

Content-based could mean user-based or product-based and the choice is depended on the mission of the business and information sparsity. A set of attributes are required to characterise the product or user, in order to have a customised and accurate prediction.   
The problem is that sometimes there are not good way to automatically attribute new product or user.

### 2. Collaborative Filtering:

As the name implies, collaborative filtering will make use of both user and product information to make the recommendation.

#### similarity

The similar product o user is defined by the similarity. However there are quite some different definition of similarity in use, e.g euclidean distance, pearson coefficient, mannhaton distance…

## Feature Creation

If you read the previous, you probably will surprise why we convert the skills to a vector just to count the occurrence. In fact vectorizing the skill is the step prepared to calculate the skill similarity between the job and me. So here I have prepared more complete list of skills:

```python
# create feature vector for skills
# a more complete skills of Data Analyst
skills=['excel','r','sql','python','tableau','d3','qlikview','hadoop','matlab','scala','sas','spss']
# add my skill if not inside the pre-define list
skills=list(set(skills+mySkill))
# split text into words for each job post
words=data['JD'].map(cleanText)
# convert the job post skills to binary vector
featureJobSkill=pd.DataFrame([convertTextFeature(text,skills) for text in words])
# convert the my skills to a binary vector
featureMySkill=convertTextFeature(mySkill,skills)
```

## Similarity Definition

My first idea is to match exactly my skills with the job skills. Either the case I have additional skill or the job requires a different skill will consider as dissimilar. In this case we can simply use euclidean distance to define the difference (denotes as d), then the similarity will be define as 1/(1+d) to convert it to a [0,1] range. The piece of idea is realised by the code below and let’s call it similarity1.

```python
# calculate the euclidean distance of two vectors
def euclideanDistance(f1,f2):
    # to make sure the two vector are having same dimension
    if len(f1)==len(f2):
        f1=pd.Series(f1)
        f2=pd.Series(f2)
        square=(f1-f2)**2
        # return square root of the sum of squares
        return round(square.sum()**0.5,2)
    else:
        return None
 
def similarity1(f1,f2):
    d=euclideanDistance(f1,f2)
    if d is None:
        return None
    else:
        return 1/(1+d)
```

However, sometimes we don’t want a minus score if we are having more skills than required, so I developed another similarity, which I call it similarity2.

```python
def similarity2(f1,f2):
    if len(f1)==len(f2):
        f1=pd.Series(f1)
        f2=pd.Series(f2)
        diff=(f1-f2)
        # f1 is job skill vector
        # f2 is my skill vector
        # ignore the case that I have more skills than required
        diff[diff&lt;0]=0
        sum_square=diff.map(lambda x: x**2).sum()
        similarity=1/(1+sum_square**0.5)
        return round(similarity,2)
    else:
        return None
```

## Assemble into a simple recommender

Now we will have a recommender requires user to input his/her skills and years of experience and we also want to have a freedom to choose output how many jobs that are the most suitable as well as the choice of the similarity function that we defined earlier.   
For those who have experience applying jobs, there is usually no harm to try the position with 2 year experience more than that you have. And I’m going to put that consideration into the recommendation as well.

```python
def jobRecommender(data,mySkill,myExperience,N=5,simlarityFunc=similarity1):
    # add experience column
    # the detail of getExperience is in previous post
    experience=data['JD'].map(getExperience)
    experience[(experience==0)|(experience&gt;20)]=None
    data['experience']=experience
 
    # create feature vector for skills
    # a more complete skills of Data Analyst
    skills=['excel','r','sql','python','tableau','d3','qlikview','hadoop','matlab','scala','sas','spss']
    # add my skill if not inside the pre-define list
    skills=list(set(skills+mySkill))
    # split text into words for each job post
    words=data['JD'].map(cleanText)
    # convert the job post skills to binary vector
    featureJobSkill=pd.DataFrame([convertTextFeature(text,skills) for text in words])
    # convert the my skills to a binary vector
    featureMySkill=convertTextFeature(mySkill,skills)
 
    # calculate similarity of the job and user in terms of skills
    data['similarity']=featureJobSkill.apply(lambda x: simlarityFunc(x,featureMySkill),axis=1)
 
    # filter the experience requirement
    # add 2 more years for the experience filter
    data=data[data.experience&lt;=(myExperience+2)].sort('similarity',ascending=False)
 
    # recommend the top N suitable jobs
    N=min(len(data),N)
    return data[['title','company','similarity']][:N]
```

Let’s the test the result. For example, I have a list of skills: excel, r, matlab, python, tableau, sql and I don’t have experience of working as a Data Analyst.

```python
myskill=['excel','sql','r','python','matlab','tableau']
myExperience=0
print jobRecommender(data,myskill,myExperience=0,N=5,simlarityFunc=similarity1)
print jobRecommender(data,myskill,myExperience=0,N=5,simlarityFunc=similarity2)
```

Using similarity function 1, I have a list of recommended job post:

|id |  title  | company |
|---|---------|---------|
|31 | Senior Data Analyst | ZALORA Group |
|52 | Technology, Prime Services Technology, Client … | Goldman Sachs |
|60 | Market Risk Reporting Analyst  | Credit Suisse |
|186| Application Support Analyst | AccorHotels |
|215| Forecasting Analyst | Xilinx |

Using similarity function 2, the recommendation is like:

|id |  title  | company |
|---|---------|---------|
|287| PricingRevenue Management Analyst (South Distr… | UPS |
|268| Application Support Analyst | AccorHotels |
|426| TRIRIGA Application Developer | IBM |
|394| (SGP-Singapore) IPB – Due Diligence Analyst | Citi |
|387| (SGP-Singapore) IT Business Implementation Ana… | Oracle |

Having done that, you can input your own skill set and experience to do some experience for a more efficient job hunting.

## The End

It comes to the part III of the Job Hunting series and I have successfully draw a period to this project, as I going to work as a Junior Data Analyst soon after a short holiday.

## Some Suggestions for future Data Analyst

### 1. Online course / nano-degree is useful

Coursera/edX/Udacity offer a lot of course in data science, data analysis and they are free if you care about the certificates. I personally took the data science specialization in coursera offered by Johns Hopkins University. The difficulty level is moderate and the programming is mainly in R, which is quite easy to pick up.

### 2. Develop a stack of skills

* Statistics:
I learnt some statistics modules in university, which are quite fundamental but indeed useful to understand many concept in machine learning and data analysis context.
* Programming:
The two most popular programming tools are python and R for data scientist and data analyst. I learnt R from R programming course in Coursera. For python language, I am still picking up and personally feel “Introduction to Computer Science and Programming in Python” offered in edX and “Design of Computer Program” in Udacity are especially helpful.
* Database and querying:
A series of database course on Stanford Online is very beneficial. The SQL quiz is somewhat challenging for beginners.
* Machine Learning:
The practical machine learning from Johns Hopkins University is a good start. The course is taught in R and mainly used caret package for ML. A more in-depth course is Stanford Machine Learning class taught by Andrew Ng. The class is taught in Matlab.

### 3. Handles on practice

* Participate Kaggle competition:
Try those kaggle 101 competitions, because there are usually a lot of tutorials, codes, demos are already available for you to get your foot wet in data analysis and machine learning in practice. And don’t get addicted to those featured level competition, because those teams are really into the money and there are not much information shared before the deadline.
* Keep doing some mini-projects:
Find some topics that interest you and try to think about the solutions in data analyst view point. This thinking process is often asked in an interview, such as “How you do think can apply machine learning in your work/business?”
* Create your own blog:
I’m trying to update a blog weekly, not only to show off my learning, but also remind me to do the some summary and review on skills. I personally feel it is very beneficial. Once the hiring manager open your blog from your resume, your chance to stand out is much higher.

### 4. Find a intern/part time job

Unlike U.S. where a lot of Tech companies hiring data analyst/scientist, Singapore’s market is not so demanding. So it’s not bad idea to get look to start with a freelance/part-time analyst or even working for free. sg.startup.job is a platform where some interesting analytical position will open by start-up companies.

### 5. Good Luck and Happy Hunting

Last and not least, decorate your resume and be confident in interview.
The rest is just luck and persistence. Happy Job Hunting and Good Luck!

