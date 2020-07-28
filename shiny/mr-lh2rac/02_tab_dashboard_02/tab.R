tab_02 <- list()

sidebar <- sidebarPanel(
  id = 'tab_02.sidebar',
  width = 3,
  numericInput('tab_02.numInput.n','Input Sample Size (N): ',
               value = 10, min = 10, max = 1000, step = 10),
  actionButton('tab_02.btn.reset','Reset')
)

mainpage <- mainPanel(
  id = 'tab_02.mainpage',
  width = 9,
  fluidRow(plotOutput('tab_02.plotOutput.chart', width = '400px', height = '400px')),
  tags$hr(),
  fluidRow(textOutput('tab_02.textOutput.pie'))
)

tab_02$ui <- tabPanel(
  title = 'Dashboard 02',
  sidebarLayout(sidebar, mainpage)
)

tab_02$server <- function(input, output) {
  
  x <- seq(-1,1,0.001)
  y <- sqrt(1 - x**2)
  circle <- data.frame(
    x = c(x,x),
    y = c(y,-y)
  )
  
  dots <- function(n){
    n <- as.integer(n)
    data.frame(
      x = runif(n, min = -1, max = 1),
      y = runif(n, min = -1, max = 1)
    )
  }
  
  in.circle <- function(dots){
    # dots is data.frame
    dots$y_boundary <- sqrt(1 - (dots$x)**2)
    mean(abs(dots$y) <= abs(dots$y_boundary))*4
  }
  
  observeEvent(input$tab_02.numInput.n, {
    
    output$tab_02.plotOutput.chart <- renderPlot({
        plot(circle, type = 'l')
        points(dots(input$tab_02.numInput.n), col = 'red')
      })
    
    output$tab_02.textOutput.pie <- renderText({
      pie <- in.circle(dots(input$tab_02.numInput.n))
      glue('estimated pi (based on N = {input$tab_02.numInput.n}) is {pie}')
    })
  })
  
  observeEvent(input$tab_02.btn.reset, {
    shinyjs::reset('tab_02.sidebar')
    shinyjs::reset('tab_02.mainpage')
  })
}