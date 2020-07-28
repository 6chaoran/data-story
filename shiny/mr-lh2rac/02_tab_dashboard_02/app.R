library(shiny)
library(shinyjs)
library(glue)

source('tab.R')

ui <- navbarPage(
  title = 'Dashboard 02',
  selected = 'Dashboard 02',
  useShinyjs(),
  tab_02$ui
)

server <- tab_02$server

shinyApp(ui = ui, server = server)