user.access <<- list(
  'liuchr' = '123456'
)

header <- tagList(
  tags$h1('this is header', 
          tags$img(src = 'logo.png', width = '200px', align = 'left', style = 'vertical-align:middle')),
  tags$hr(class = 'hline')
)

footer <- tagList(
  tags$hr(class = 'hline'),
  tags$p('Munich Re Singapore Branch')
)
