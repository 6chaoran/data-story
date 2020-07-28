Bay Area Bike Share Analysis
================
Liu Chaoran
10/07/2017

Excutive Summary
----------------

I'm interested to look at the Bike usage in terms of duration and trip count. A reasonable assumption is to perform a perodic maintainance/check-up for bikes in order to provide satisifing customer service. This report of analysis will focus on bike usage and suggest when and where to recycle the bike and put back the bike for operation.

Overview Statistics
-------------------

-   the total number of active bikes: **681**
-   the average bike usage duration: **4762** mins
-   the average bike usage trip: **252**

Analysis of Bike Usage
----------------------

There are about 600 bikes running in Bay Area, covers San Fransico, Palo Alto, Mountain View and San Jose.

### 1.distribution of number of active bikes is fairly stable

``` r
p1 <- bike_num_month %>% ggplot(aes(month,bike_num))+geom_bar(stat = 'identity',fill = 'lightblue')+xlab("")
p2 <- bike_num_wday %>% ggplot(aes(wday,bike_num))+geom_bar(stat = 'identity',fill = 'lightblue')+xlab("")
p3 <- bike_num_day %>% ggplot(aes(day,bike_num))+geom_bar(stat = 'identity',fill = 'lightblue')+xlab("")
multiplot(p1,p2,p3,cols = 1)
```

![](2_Analysis_files/figure-markdown_github/distr%20number%20of%20active%20bikes-1.png)

### 2.bike usage show binomial distribution

This is partly contributed by different subscriber type: customer and subscriber.

``` r
p1 <- bike_usage %>% ggplot(aes(trip_num)) + geom_histogram(fill = 'lightblue',bins = 30)
p2 <- bike_usage %>% ggplot(aes(duration)) + geom_histogram(fill = 'lightblue',bins = 30)
multiplot(p1,p2,cols = 2)
```

![](2_Analysis_files/figure-markdown_github/distr%20bikes%20usage-1.png)

Another reason to cause the binomial distribution usage is the geo-location. Shown in below chart, from bottom-left to top-right are : San Jose, Mountain View, Palo Alto and San Fransico.
The bike in San Fransico are much highlier used than the rest area.

``` r
stn2stn_usage %>% ggplot(aes(Start_Terminal,End_Terminal))+geom_tile(aes(fill = bike_usage_trip))+
  scale_fill_gradient(low = "lightblue",high = 'darkblue')
```

![](2_Analysis_files/figure-markdown_github/distr%20bikes%20usage%20by%20station-1.png)

### 3. bike usage distribution by time

Let's assume we will run the bike recycle and check-up once a week. Therefore, we need find a good time to perform this activity. By examining the bike usage, we will find a timeslot that bike usage is low.
Subscriber usually ride on weekdays for commuting, and their trips are generally frequent but short.
However customers usually ride more on weekends and longer trip.

``` r
p1 <- bike_usage_wday %>% ggplot(aes(wday,trip_num))+geom_boxplot() + geom_jitter(alpha = 0.2) + facet_grid(~subs)+xlab("")
p2 <- bike_usage_wday %>% ggplot(aes(wday,duration))+geom_boxplot() + geom_jitter(alpha = 0.2) + facet_grid(~subs)+xlab("")
multiplot(p1,p2,cols = 2)
```

![](2_Analysis_files/figure-markdown_github/distr%20bikes%20usage%20by%20time-1.png)

### 4. bike periodic check-up

Looking at the overall distribution, I assmue a citeria of bike check-up: trip\_num &gt; 500 or duration\_total &gt; 10000.

-   The according number of bikes need to check-up: **42**
-   proportion of bikes need to check-up: **6.2%**
-   bike check-up is suggested to perform on **weekends** due to low expected usage

Looking at the indiviual bike usage distrution, a few more things are needed for check-up and put-back practice:

-   usage distribution in day of week: find the low usage day to check-up
-   last parked station: the location to pick the bike
-   highly serviced station: top 3 stations the bike served, which will be the location to put back for serivce.

The detailed information will be populated in dashboard for easy look-up.

Take the **bike\# 137** for example, bike duration is &gt; 10000 min, (about keep running for 7 days), which need a perodic check-up. The low usage weekday is **Sunday**, operators will pick the bike\# 137 from **Commercial at Montgomery** on **Sunday** and put back to service at station **San Francisco Caltrain (Townsend at 4th)**.

``` r
kable(data.frame(bike_db %>% filter(needCheckUp==1) %>% head(3)))
```

| Bike\_. |  trip\_num|  duration|  usage\_Sun|  usage\_Mon|  usage\_Tues|  usage\_Wed|  usage\_Thurs|  usage\_Fri|  usage\_Sat| lastStation              | serviceStation                           |  needCheckUp|
|:--------|----------:|---------:|-----------:|-----------:|------------:|-----------:|-------------:|-----------:|-----------:|:-------------------------|:-----------------------------------------|------------:|
| 137     |        377|  10322.48|          28|          46|           64|          79|            66|          65|          29| Commercial at Montgomery | San Francisco Caltrain (Townsend at 4th) |            1|
| 251     |         45|  13024.27|          12|           4|            7|           5|             6|           7|           4| Adobe on Almaden         | San Jose Diridon Caltrain Station        |            1|
| 275     |        451|  10251.38|          25|          66|           94|         100|            68|          68|          30| 2nd at Townsend          | San Francisco Caltrain (Townsend at 4th) |            1|

Findings/Conclusions
--------------------

1.  customers and subscribers riding behaviour differ a lot. customers ride less frequent but long trip, whie the subscribers are the opposite. The different pricing strategy should be adapt for both, such as charge-by-time for customers and flat-charge-per-ride for subscribers.
2.  There are about 6.2% bikes need to go through check-up based on make-up check-up citeria. The best check-up time should be on weekends.
