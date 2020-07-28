library(shiny)
library(shinyjs)
library(glue)

source('tab.R')

ui <- navbarPage(
  title = 'Dashboard 01',
  selected = 'Dashboard 01',
  useShinyjs(),
  tab_01$ui
)

server <- tab_01$server

shinyApp(ui = ui, server = server)