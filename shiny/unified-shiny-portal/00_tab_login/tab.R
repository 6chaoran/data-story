# define a list to wrap ui/server for login tab
tab_login <- list() 

# define ui as tabPanel,
# so that it can be easily added to navbarPage of main shiny program
tab_login$ui <- tabPanel(
  title = 'Login', # name of the tab
  # hide the welcome message at the first place
  shinyjs::hidden(tags$div(
    id = 'tab_login.welcome_div',
    class = 'login-text', 
    textOutput('tab_login.welcome_text', container = tags$h2))
  )
)

# define the ui of the login dialog box
login_dialog <- modalDialog(
  title = 'Login to continue',
  footer = actionButton('tab_login.login','Login'),
  textInput('tab_login.username','Username'),
  passwordInput('tab_login.password','Password'),
  tags$div(class = 'warn-text',textOutput('tab_login.login_msg'))
)

# define the backend of login tab
## show dialog box when stat up
## when login failed, update the warning message
## when login succeeded, remove the dialog box and show the welcome message
tab_login$server <- function(input, output) {
  
  # show login dialog box when initiated
  showModal(login_dialog)
  
  observeEvent(input$tab_login.login, {
    username <- input$tab_login.username
    password <- input$tab_login.password
    
    # validate login credentials
    if(username %in% names(user.access)) {
      if(password == user.access[[username]]) {
        # succesfully log in
        removeModal() # remove login dialog
        output$tab_login.welcome_text <- renderText(glue('welcome, {username}'))
        shinyjs::show('tab_login.welcome_div') # show welcome message
      } else {
        # password incorrect, show the warning message
        # warning message disappear in 1 sec
        output$tab_login.login_msg <- renderText('Incorrect Password')
        shinyjs::show('tab_login.login_msg')
        shinyjs::delay(1000, hide('tab_login.login_msg'))
      }
    } else {
      # username not found, show the warning message
      # warning message disappear in 1 sec
      output$tab_login.login_msg <- renderText('Username Not Found')
      shinyjs::show('tab_login.login_msg')
      shinyjs::delay(1000, hide('tab_login.login_msg'))
    }
  })
}