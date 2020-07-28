tab_01 <- list()

sidebar <- sidebarPanel(
  id = 'tab_01.sidebar',
  width = 3,
  numericInput('tab_01.numInput.n', 'Input Sample Size (N): ', 
                 value = 50, min = 50, max = 1000, step = 50),
  actionButton('tab_01.reset','Reset')
)

mainpage <- mainPanel(
  id = 'tab_01.mainpage',
  width = 9,
  plotOutput('tab_01.plotOutput.hist')
)

tab_01$ui <- tabPanel(
  title = 'Dashboard 01',
  sidebarLayout(sidebar, mainpage)
)

tab_01$server <- function(input, output) {
  
  observeEvent(input$tab_01.numInput.n, {
    output$tab_01.plotOutput.hist <- renderPlot(
      hist(rnorm(input$tab_01.numInput.n), 
           main = glue('Histogram with N = {input$tab_01.numInput.n}'),
           xlab = 'N'))
  })
  
  observeEvent(input$tab_01.reset, {
    shinyjs::reset('tab_01.sidebar')
    shinyjs::reset('tab_01.mainpage')
  })
}