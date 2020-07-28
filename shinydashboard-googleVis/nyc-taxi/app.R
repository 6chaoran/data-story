library(shiny)
library(shinydashboard)
source('nyc-taxi.R')
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
server <- function(input, output) {
  
  # every time the action button clicked, the df.sample is refreshed
  df.sample <- eventReactive(input$i.btn,{
    df %>% sample_n(size = input$i.sampleSize)
  })
  
  # render my plots
  output$my.barchart <- renderGvis(my.barchart(df.sample()))
  output$my.piechart <- renderGvis(my.piechart(df.sample()))
  output$my.table <- renderGvis(my.table(df.sample()))
  output$my.map <- renderLeaflet(my.map(df.sample()))
  
  
}

# Run the application 
shinyApp(ui = ui, server = server)

