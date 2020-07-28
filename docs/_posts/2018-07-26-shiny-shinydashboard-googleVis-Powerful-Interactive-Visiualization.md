---
title: "Shiny + shinydashboard + googleVis = Powerful Interactive Visiualization"
date: 2018-07-26 23:02:01 +0800
categories: 
  - visualization
tags:
  - shiny
  - R
toc: true
toc_sticky: true
---
  
If you are a data scientist, who spent several weeks on developing a fantanstic model, you'd like to have an equally awesome way to visualize and demo your results. For R users, ggplots are good option, but no longer sufficient. R-shiny + shinydashboard + googleVis could be a wonderful combination for a quick demo application.
For the purpose of illustration, I just downloaded a random sample data `test.csv` from kaggle's latest competitions:
[https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data)   

![shiny](https://github.com/6chaoran/data-story/raw/master/shinydashboard-googleVis/shiny.PNG)

The R shiny app is available here: [https://6chaoran.shinyapps.io/nyc-taxi/](https://6chaoran.shinyapps.io/nyc-taxi/)

### R Shiny
R Shiny is an interactive tool for R users to seemlessly integrate R eco-system and quickly develop a demo app.   
Some key concepts of Shiny:
1) **UI**: a web page connected with backend R session. It's basically written using HTML/Javascript/CSS. By default, Shiny use the `fluidPage` template, so it is written as `ui <- fluidPage()`.
2) __Server__: the backend R session, which does all the data manipulation and calculation. The model prediction part should sit here as well. The server can be a function of `input`, `output` and `session`, so it is written as `server <- function(input, output, session){...}`. `input` is a list of variables pass from UI, while `output` is a list of varialbes rendered from Server and ready to display in UI.
3) __App__: a shiny app, comprises `ui` and `server`. It can be two seperate files named `ui.R` and `server.R`, or a single file named `app.R`, consisting of ui function, server function and `shinyApp(ui, server)`

reference: [https://shiny.rstudio.com/tutorial/](https://shiny.rstudio.com/tutorial/)

### shinydashboard package
shinydashboard provide neat and nice interface, with customizable header, sidebar and body. I usually just need to define my title in header, list the tabs or selection panel in sidebar, and at last assemble all the tables and charts into the body section. Make use of `fluidRow` and `column` function to align your components in body.    

we can use following code to build the framework of the dashboard:

![shinydashboard](https://github.com/6chaoran/data-story/raw/master/shinydashboard-googleVis/shinydashboard.PNG)

```r
library(shiny)
library(shinydashboard)
# ui
header <- dashboardHeader(title = 'NYC Taxi Explore')
sidebar <- dashboardSidebar(
  numericInput('i.sampleSize', 'Choose the size of your sample:', min = 10, max = 5000, step = 50,value = 100),
  actionButton('i.btn', 'Refresh!', icon = icon('paper-plane'))
)
body <- dashboardBody(
  
  fluidRow(
    column(width = 12,
           HTML("<b>Instruction:</b><br>
                1. set your size for data sampling, from 10 to 5000.<br>
                2. hit refresh! button to get your charts.<br>
                data source: https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data<br>"))
  ),
  tags$hr(),
  # first row
  fluidRow(
    column(width = 6, # my map
           leafletOutput('my.map')),
    column(width = 6, # my pie chart
           htmlOutput('my.piechart'))
  ),
  
  # second row
  tags$hr(),
  fluidRow(
    column(width = 6, # my bar-chart
           htmlOutput('my.barchart')),
    column(width = 6, # my table
           HTML("<br>"),
           htmlOutput('my.table'))
  )
)

ui <- dashboardPage(header,sidebar,body)
# server
server <- function(input, output) {}
# Run the application 
shinyApp(ui = ui, server = server)
```

reference: [https://rstudio.github.io/shinydashboard/get_started.html](https://rstudio.github.io/shinydashboard/get_started.html)

### googleVis for R
Google Visualization is developed using Javascript and R package is available as `googleVis` in CRAN, with some limitation. Simple bar-chart, line-chart, tables are easily construsted and be able to meet our general needs. 
Here is the exmample plots using googleVis: https://cran.r-project.org/web/packages/googleVis/vignettes/googleVis_examples.html
Detail customerization of the charts need look up the documentation of google's javascript API and supply the setting as JSON format text to the `options` in R (e.g. `options = list(hAxis = "{textStyle:{fontSize:12},format:'percent',minValue:0,maxValue:1}"`)  

For example, a simple donut chart can be constructed using following code:

![pie-chart](https://github.com/6chaoran/data-story/raw/master/shinydashboard-googleVis/googleVis.PNG)

```r
library(dplyr)
library(data.table)

df <- fread('test.csv')

library(googleVis)
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
                                height = "400")) %>%
    plot()
```

Google Vis Official Documenation: [https://cran.r-project.org/web/packages/googleVis/vignettes/googleVis_examples.html](https://cran.r-project.org/web/packages/googleVis/vignettes/googleVis_examples.html)

### leaflet for R
googleVis is good enough for most of charts, except for maps, because maps in googleVis is replied on googleMap API, which is not free of charge. Leaflet for R is an good alternative.

![leaflet](https://github.com/6chaoran/data-story/raw/master/shinydashboard-googleVis/leaflet.PNG)

```r
library(leaflet)
df %>% 
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
```
reference: [https://rstudio.github.io/leaflet/](https://rstudio.github.io/leaflet/)

### Put Together
I make 4 functions to plot 4 charts respectives and defined in a seperate R file, [nyc-taxi.R](https://github.com/6chaoran/data-story/blob/master/shinydashboard-googleVis/nyc-taxi/nyc-taxi.R), which is then loaded in our shiny app, [app.R](https://github.com/6chaoran/data-story/blob/master/shinydashboard-googleVis/nyc-taxi/app.R).   
If you'd like to replicate the result, please feel free to click the links above to view the [complete R code](https://github.com/6chaoran/data-story/tree/master/shinydashboard-googleVis/nyc-taxi).
