---
title: "Getting Started With Tableau"
author: "Liu Chaoran"
date: "10 August 2015"
output: html_document
---
## Intro to Tableau
Aspired by the course 'Data Visualization' offered by University of Illinois on Cousera, I have worked on the interactive data visualization using Tableau. There is a free version of [Tableau Public](https://public.tableau.com/s/) is available and you can upload the visualization online for sharing.    
Tableau is one of the Business Intelligence tools that makes it easier to do with aesthetic chart plotting and interactive report generating. There are 3 main components used in Tableau: Worksheet, Dashboard and Story.    
* __Worksheets__ are single chart or plot
* __Dashboard__ is a single page can compose with mupliple charts or plots
* __Story__ is like powerpoint in MS Office, which put a series of pages of charts in sequence.    

There are two kinds of charts that interest me: smart map and bubble chart.   
I'm going to make use of the two chart to visual the U.S. Weather data set.

## Description of the data set
The data is taken by U.S. Weather Serivce, the link of data set is available [here](https://d396qusza40orc.cloudfront.net/repdata%2Fdata%2FStormData.csv.bz2). The comphrensive description of the data is in [here](https://d396qusza40orc.cloudfront.net/repdata%2Fpeer2_doc%2Fpd01016005curr.pdf).   

## Data manipulation using R
As Tableau is not ideal for data munging, I would like to do the first-hand pre-processing with R. 


```r
head(data[c("EVTYPE",   "LATITUDE"  , "LONGITUDE" , "STATE" ,"BGN_DATE","FATALITIES","INJURIES","PROPDMG","PROPDMGEXP" )])
```

```
   EVTYPE LATITUDE LONGITUDE STATE           BGN_DATE FATALITIES INJURIES
1 TORNADO     3040      8812    AL  4/18/1950 0:00:00          0       15
2 TORNADO     3042      8755    AL  4/18/1950 0:00:00          0        0
3 TORNADO     3340      8742    AL  2/20/1951 0:00:00          0        2
4 TORNADO     3458      8626    AL   6/8/1951 0:00:00          0        2
5 TORNADO     3412      8642    AL 11/15/1951 0:00:00          0        2
6 TORNADO     3450      8748    AL 11/15/1951 0:00:00          0        6
  PROPDMG PROPDMGEXP
1    25.0          K
2     2.5          K
3    25.0          K
4     2.5          K
5     2.5          K
6     2.5          K
```

### Formatting variable Year, Longitude, Latitude

```r
data$BGN_DATE<-as.Date(data$BGN_DATE,format='%m/%d/%Y %H:%M:%S')
library(lubridate)
data$Year<-year(data$BGN_DATE)
data$LONGITUDE<-(-data$LONGITUDE/100)
data$LATITUDE<-data$LATITUDE/100
head(data[c("EVTYPE",   "LATITUDE"  , "LONGITUDE" , "STATE" ,"Year","FATALITIES","INJURIES","PROPDMG","PROPDMGEXP" )])
```

```
   EVTYPE LATITUDE LONGITUDE STATE Year FATALITIES INJURIES PROPDMG
1 TORNADO    30.40    -88.12    AL 1950          0       15    25.0
2 TORNADO    30.42    -87.55    AL 1950          0        0     2.5
3 TORNADO    33.40    -87.42    AL 1951          0        2    25.0
4 TORNADO    34.58    -86.26    AL 1951          0        2     2.5
5 TORNADO    34.12    -86.42    AL 1951          0        2     2.5
6 TORNADO    34.50    -87.48    AL 1951          0        6     2.5
  PROPDMGEXP
1          K
2          K
3          K
4          K
5          K
6          K
```

### Creating Economic Loss = Property Loss + Crop Loss

```r
data$PROPDMGEXP<-transformEXP(data$PROPDMGEXP)
data$CROPDMGEXP<-transformEXP(data$CROPDMGEXP)
data$economicLoss<-with(data,(PROPDMG*(10**PROPDMGEXP)+CROPDMG*(10**CROPDMGEXP)))
```
### Creating Health Impact = Fatalities + Injuries

```r
data$healthImpact<-data$FATALITIES+data$INJURIES
head(data[c("EVTYPE",   "LATITUDE"  , "LONGITUDE" , "STATE" ,"BGN_DATE","healthImpact","economicLoss" )])
```

```
   EVTYPE LATITUDE LONGITUDE STATE   BGN_DATE healthImpact economicLoss
1 TORNADO    30.40    -88.12    AL 1950-04-18           15        25000
2 TORNADO    30.42    -87.55    AL 1950-04-18            0         2500
3 TORNADO    33.40    -87.42    AL 1951-02-20            2        25000
4 TORNADO    34.58    -86.26    AL 1951-06-08            2         2500
5 TORNADO    34.12    -86.42    AL 1951-11-15            2         2500
6 TORNADO    34.50    -87.48    AL 1951-11-15            6         2500
```

### Subset the top 5 weather
Too many EVTYPE, let's look at the top 5 in frequency, after removing the disasters without impact or loss.

```r
data<-data[data$economicLoss!=0 | data$healthImpact!=0,]
freqEVTYPE<-data.frame(table(data$EVTYPE))
freqEVTYPE<-sortDF(freqEVTYPE,freqEVTYPE$Freq)
head(freqEVTYPE)
```

```
                 Var1  Freq
423         TSTM WIND 63234
364 THUNDERSTORM WIND 43655
407           TORNADO 39944
134              HAIL 26130
73        FLASH FLOOD 20967
258         LIGHTNING 13293
```

```r
top5<-as.character(freqEVTYPE[1:5,1])
data<-data[data$EVTYPE %in% top5,]
```

### Select the variables we need

```r
var<-c("EVTYPE",   "LATITUDE"  , "LONGITUDE" , "STATE" ,"Year","healthImpact","economicLoss" )
data<-data[,var]
```

### Output the data.csv

```r
write.csv(data,'data.csv',row.names=F)
```
