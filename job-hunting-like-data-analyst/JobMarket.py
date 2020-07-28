import pandas as pd
csv_file='/Users/6chaoran/Desktop/sgJobs.csv'
# pandas csv loading, similar with read_csv in R readr package
data=pd.read_csv(csv_file)

#====Plot Job Market=====
#freqPlot(data,'company')
#freqPlot(data,'title')
%matplotlib inline
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

# load the regular experession package
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

def convertTextFeature(text,feature):
	featureVector={}
	for f in feature:
		if f in text:
			featureVector[f]=1
		else:
			featureVector[f]=0
	return featureVector

# load the regular experession package
import re
def getExperience(text):
	# find all the '...years' patterns in the job description
	years=re.findall('....years',text)
	def yearToNumber(years):
		try:
			return int(re.sub('[^0-9]','',years))
		except:
			return None
	years=map(yearToNumber,years)
	if len(years)>0:
		return max(years)
	else:
		return None

## Skills
# pre-defined skills 
skills=['excel','r','sql','python','tableau','d3','qlikview','hadoop','matlab','scala','sas','spss']

# convert JD to a bag of words
words=data['JD'].map(cleanText)

# vectorize the skills in data record
featureSkill=pd.DataFrame([convertTextFeature(text,skills) for text in words])

##plot the skill requirement
# sum the skill across the row
skillList=featureSkill.sum(axis=0)
skillList.sort()
skillList.plot(kind='barh',title='the skill requirement for a data analyst')

## Experience
## get the experience
experience=data['JD'].map(getExperience)
experience[(experience==0)|(experience>20)]=None
data['experience']=experience

freqPlot(data,'experience',title='experience requirement for Data Analyst')

