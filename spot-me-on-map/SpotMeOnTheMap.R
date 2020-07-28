##Taxi in Porto
##loading data
setwd("/Users/6chaoran/Dropbox/kaggle/taxi")
library(readr)
test=read_csv('test.csv.zip')
polyline=test$POLYLINE

##transform POLYLINE string
getCoord=function(x){
	x=gsub('[/[]','',x)
	x=gsub(']','',x)
	x=strsplit(x,',')[[1]]
	n=length(x)
	lon=as.numeric(x[seq(1,n,2)])
	lat=as.numeric(x[seq(2,n,2)])
	df=data.frame(lon=lon,lat=lat)
	df$status='moving'
	df$status[1]='pickup'
	df$status[nrow(df)]='dropoff'
	return(df)
}

##create a matrix of coordinate of the taxi location
loc=NULL
for (i in 1:length(polyline)){
	loc=rbind(loc,getCoord(polyline[i]))
}

##kmeans clustering for pickup and dropoff
dropoff=loc[loc$status=='dropoff',c('lon','lat')]
kmeans=kmeans(dropoff,4)
dropoff_centers=data.frame(kmeans$centers)
dropoff_centers$status='dropoff'
pickup=loc[loc$status=='pickup',c('lon','lat')]
kmeans=kmeans(pickup,4)
pickup_centers=data.frame(kmeans$centers)
pickup_centers$status='pickup'
centers=rbind(pickup_centers,dropoff_centers)

##plot taxi trajectory on google map
library(ggmap)
map=get_map(location=c(lon=median(loc$lon),lat=median(loc$lat)),
	maptype='roadmap',zoom=13)
plot_taxi=ggmap(map)+labs(x='Longitude',y='Latitude')+
geom_jitter(aes(lon,lat,colour=factor(status)),data=loc[loc$status=='moving',],alpha=0.4)+
geom_point(aes(lon,lat,colour=factor(status)),data=centers,size=10,alpha=0.6)+
geom_jitter(aes(lon,lat,colour=factor(status)),data=loc[loc$status!='moving',],alpha=0.8)



##Crime in San Fansisco
##loading data
setwd("~/Dropbox/kaggle/San Fransico Crime Classification")
library(readr)
data=read_csv('train.csv.zip',col_types=list(Dates=col_datetime(format='%Y-%m-%d %H:%M:%S')))
library(lubridate)
data$Year=year(data$Dates)

##get data from Year 2014 only
data_2014=data[data$Year==2014,]
data0=data[1,]
Crime=data.frame(table(data_2014$Category))
Crime=Crime[order(Crime$Freq,decreasing=T),]
Top3=Crime[c(1,4,5),1]


library(ggmap)
map=get_map(location='San Fransico',maptype='roadmap',zoom=12)
plot_crime=ggmap(map)+geom_jitter(aes(X,Y,colour=Category),
	data=data_2014[data_2014$Category %in% Top3,],alpha=0.2)+geom_jitter(aes(X,Y,colour=Category),
	data=data0[data0$Category %in% Top3,],alpha=1)+labs(x='Longitude',y='Latitude')

setwd("~/Desktop/DataStory/spot-me-on-map")
library(knitr)
knit2html('README.Rmd')
