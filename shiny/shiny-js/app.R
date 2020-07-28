library(shiny)
library(markdown)

shiny.logo <- "https://raw.githubusercontent.com/rstudio/shiny/master/man/figures/logo.png"
js.logo <- "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/JavaScript-logo.png/240px-JavaScript-logo.png"
d3.logo <- "https://raw.githubusercontent.com/d3/d3-logo/master/d3.png"

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Shiny/JavaScript Integration"),
    tags$script(src="https://d3js.org/d3.v5.min.js"),
    tags$script(src="https://unpkg.com/d3-array@1"),
    tags$script(src="https://unpkg.com/d3-collection@1"),
    tags$script(src="https://unpkg.com/d3-path@1"),
    tags$script(src="https://unpkg.com/d3-shape@1"),
    tags$script(src="https://unpkg.com/d3-sankey@0"),
    includeCSS("./css/styles.css"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(width = 3,
                     
            # histogram controls
            sliderInput("bins",
                        "Number of bins:",
                        min = 1,
                        max = 50,
                        value = 30),
            tags$button('set bins to 100', onclick= 'setBinsTo100()'),
            tags$p("click button above to trigger JavaScript code", 
                   tags$pre('Shiny.setInputValue(\"bins\", 100);')),
            tags$script("
            function setBinsTo100(){
                Shiny.setInputValue('bins', 100);
                alert('number of bins is set to 100 !');
            }"),
            tags$br(),
            tags$br(),
            
            # d3 sankey chart controls
            selectInput('sankeyAlign','sankeyAlign: ',
                        choices = c(Left = 'left',
                                    Right = 'right',
                                    Justified = 'justify',
                                    Center = 'center'),
                        selected = 'justify'),
            selectInput('edgeColor','edgeColor: ',
                        choices = c(`Color by input` = 'input',
                                    `Color by ouptut` = 'output',
                                    `No color` = 'none'),
                        selected = 'none'),
            tags$button(onclick="sankeyShow()", "show sankey"),
            tags$button(onclick="sankeyHide()", id="btnSankeyHide", "hide sankey"),
            tags$p("click the button above to generate d3 sankey plot")

        ),

        # Show a plot of the generated distribution
        mainPanel(
            
            # shiny/js communication section
            tags$h3(tags$image(style="width:30px; align='left'",
                               src=shiny.logo),
                    tags$image(style="width:30px; align='left'",
                               src=js.logo),
                    'Communicate between Shiny and JavaScript'),
            # parse some information from markdown
            includeMarkdown("note.md"),
            fluidRow(
                column(width = 10, plotOutput("distPlot")),
                column(width = 2,  tags$div(id="binsHistory", tags$b("History of Bins")))
            ),
            
            # d3 section
            tags$h3(tags$image(style="width:30px; align='left'",
                              src=d3.logo),
                   'Test d3 visualization in shiny'),
            tags$p('click the [show sankey] button on the sidebar to show the plot'), 
            # placeholder for sankey chart
            htmlOutput("chartSankey"),
            # JS code for sankey chart
            includeScript("./js/sankey.js"), 
            # JS code to receive message from shiny session
            tags$script(HTML("
            Shiny.addCustomMessageHandler('shinyMessageOut', 
            function(message) {
                time = message.time;
                message = message.bins;
                domHistory = document.getElementById('binsHistory');
                msg = document.createElement('p');
                msg.innerText = time + ': ' + message;
                msg.style.color = (+message) <= 50 ? 'black' : 'red';
                domHistory.append(msg);
            });"))
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output, session) {

    output$distPlot <- renderPlot({
        # generate bins based on input$bins from ui.R
        x    <- faithful[, 2]
        bins <- seq(min(x), max(x), length.out = input$bins + 1)

        # draw the histogram with the specified number of bins
        hist(x, breaks = bins, col = 'darkgray', border = 'white')
    })
    
    # send over the time/bins as the list in R / json in JS
    # whenever the input$bins changes
    observeEvent(input$bins, {
        session$sendCustomMessage('shinyMessageOut',  
                                  list(bins = input$bins, 
                                       time = Sys.time()))
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
