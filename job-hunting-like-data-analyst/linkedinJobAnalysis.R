FreqPlot<-function(df,col,fill='#33CCFF',top=10){
	data=data.frame(table(df[col]))
	data=data[order(data$Freq,decreasing=T),]
	data=data[1:top,]
	data=transform(data,Var1=reorder(Var1,Freq))
	require(ggplot2)
	return (ggplot(data,aes(Var1,Freq))+geom_bar(stat='identity',fill=fill)+coord_flip()+theme_bw())
}


## load the csv file that was scraped from linkedin
library(readr)
cn.jobs<-read_csv('cnJobs.csv')
## filter job title, which should contain key words: data and analyst
## some job title is not specific to analyst, instead they state, analytics or analysis
## so I decide to use 'analy' to include more relevent jobs
find.data.analyst<-function(title){
	data=grepl('[Dd]ata',title)
	analy=grepl('[Aa]naly',title)
	return (data&analy)
} 

true.jobs<-sapply(cn.jobs$title,find.data.analyst)
true.jobs<-cn.jobs[true.jobs,]

## clean the location
### take everything before ,CN
true.jobs$location<-sapply(true.jobs$location,function(x) strsplit(x,',')[[1]][1])
### get rid of special character
true.jobs$location<-sapply(true.jobs$location,function(x) gsub('[^a-zA-Z]','',x))
### get rid of 'City','China'
true.jobs$location<-sapply(true.jobs$location,function(x) gsub('City','',x))
true.jobs$location<-sapply(true.jobs$location,function(x) gsub('China','',x))

## Plot the barplot
FreqPlot(true.jobs,'location')
FreqPlot(true.jobs,'company')