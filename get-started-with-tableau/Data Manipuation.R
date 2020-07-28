## loading data from
library(readr)
data<-read_csv('data.csv.bz2',col_types='dcccdcccdccccdcdccdddddddcdccccddddcd')

## Formatting variable Year, Longitude, Latitude
data$BGN_DATE<-as.Date(data$BGN_DATE,format='%m/%d/%Y %H:%M:%S')
library(lubridate)
data$Year<-year(data$BGN_DATE)
data$LONGITUDE<-(-data$LONGITUDE/100)
data$LATITUDE<-data$LATITUDE/100

## Creating the features we want to study:

### Economic Loss = Property Loss + Crop Loss
data$PROPDMGEXP<-transformEXP(data$PROPDMGEXP)
data$CROPDMGEXP<-transformEXP(data$CROPDMGEXP)
data$economicLoss<-with(data,(PROPDMG*(10**PROPDMGEXP)+CROPDMG*(10**CROPDMGEXP)))

## Health Impact = Fatalities + Injuries
data$healthImpact<-data$FATALITIES+data$INJURIES


## Too many EVTYPE, let's look at the top 5 in frequency:
## remove the disasters without impact or loss:
data<-data[data$economicLoss!=0 | data$healthImpact!=0,]
freqEVTYPE<-data.frame(table(data$EVTYPE))
freqEVTYPE<-sortDF(freqEVTYPE,freqEVTYPE$Freq)
top5<-as.character(freqEVTYPE[1:5,1])
data<-data[data$EVTYPE %in% top5,]

## select the variables we need
var<-c("EVTYPE",   "LATITUDE"  , "LONGITUDE" , "STATE" ,"Year","healthImpact","economicLoss" )
data<-data[,var]


## output the data.csv
write.csv(data,'data.csv',row.names=F)


## function sort data descendingly
sortDF<-function(df,by){
  df<-df[order(by,decreasing = T),]
  return (df)
}

## function transform EXP to integers
transformEXP<-function(var){
  if (class(var)!= 'character') var=sapply(var,as.character)
  x=tolower(var)
  x[x=='b']<-9
  x[x=='h']<-2
  x[x=='k']<-3
  x[x=='m']<-6
  x[x %in% c("","-","?","+")]<-0
  x=as.integer(x)
  return (x)
}