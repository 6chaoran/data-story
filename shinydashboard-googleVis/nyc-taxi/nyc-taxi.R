library(dplyr)
library(data.table)

df <- fread('test.csv')

library(leaflet)
my.map <- function(df){
  
  # make a simple kmeans clustering
  kms.pickup <- df %>% 
    select(pickup_longitude, pickup_latitude) %>%
    kmeans(2)
  
  # plot the base map using leaflet
  base.plot <- df %>% 
    leaflet() %>%       # call leaflet
    addTiles() %>%      # add map
    # plot pickup locations
    addCircles(lng = ~pickup_longitude, # use ~ to map the variable
               lat = ~pickup_latitude, 
               color = 'red',
               opacity = 0.5,
               popup  = ~paste('key',key,'<br>',  # popup support HTML formating, <br> to break line
                               'pickup time',pickup_datetime)) %>%
    # plot dropoff location
    addCircles(lng = ~dropoff_longitude, 
               lat = ~dropoff_latitude, 
               color = 'green', 
               opacity = 0.5, 
               label = ~key) %>%
    # center the map to the middle
    setView(lng =mean(df$pickup_longitude), 
            lat = mean(df$pickup_latitude), 
            zoom = 12)
  
  # add clusters
  base.plot %>%
    addMarkers(lng = kms.pickup$centers[,'pickup_longitude'], 
               lat = kms.pickup$centers[,'pickup_latitude'],
               label = 'pickup clusters')
}

library(googleVis)
my.barchart <- function(df){
  df %>%
    mutate(passenger_count = as.character(passenger_count)) %>% # change type from int to char, make sure barplot to be correct format
    group_by(passenger_count) %>%
    summarise(n_trip = n()) %>%
    gvisBarChart(xvar = 'passenger_count', 
                 yvar = 'n_trip',
                 options = list(animation = "{startup:true, duration:1000}", # refer to google Vis document, pass in json options
                                legend = "{position:'bottom'}",
                                backgroundColor = "#edeff2",
                                title = "Number of Trips by Passenger Count"))
}

library(lubridate)
my.piechart <- function(df){
  df %>%
    mutate(pickup_hour = hour(pickup_datetime)) %>%
    mutate(pickup_time = case_when(
      between(pickup_hour,3,6) ~ 'early morning',
      between(pickup_hour,7,10) ~'morning',
      between(pickup_hour,11,14) ~'noon',
      between(pickup_hour,15,18) ~'afternoon',
      between(pickup_hour,19,22) ~'evening',
      TRUE ~ 'midnight'
    )) %>%
    group_by(pickup_time) %>%
    summarise(n_trip = n()) %>%
    gvisPieChart(labelvar = 'pickhour', 
                 numvar = 'n_trip', 
                 options = list(title = 'Pickup Time Distribution',
                                legend = "{position:'right'}",
                                pieHole = 0.5,
                                slices = "{0:{offset:0.2}}",
                                backgroundColor = "#edeff2",
                                width = "600",
                                height = "400"))
}


my.table <- function(df){
  df %>%
    mutate(passenger_count = as.character(passenger_count)) %>% 
    group_by(passenger_count) %>%
    summarise(n_trip = n()) -> res
  
  res %>%
    mutate(dummy_boolean = sample(c(T,F),nrow(res),replace = T)) %>%
    gvisTable(formats = list(n_trip = '#,###'),
              options = list(width = "400"))
}



