library(shiny)
library(shinyjs)
library(glue)

source('tab.R') # load tab_login ui/server

ui <- navbarPage(
  title = 'Login',
  selected = 'Login',
  useShinyjs(), # initiate javascript
  tab_login$ui # append defined ui
)

server <- tab_login$server # assigne defined server

shinyApp(ui = ui, server = server)
