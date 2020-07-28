# Enhance Shiny with Javascript

## Code Structure

```
shiny-js
│   app.R  			 # shiny app code
│   index.html   	 # index html for html/js/css testing
│   note.md  		 # some markdown writeup in between shiny
│   README.md
├───css
│       styles.css   # css file to format shiny
├───data
│       energy.csv   # data used for html/js/css testing
├───js
│       sankey.js    # d3.js code for sankey
└───www
        energy.csv   # data need to be placed in www directory for shiny
```

## Live Demo on ShinyApps.IO
https://6chaoran.shinyapps.io/shiny_js/

## Introduction

Shiny is invented to quickly build simple web-app for those who don't have deep experience in JavaScript or HTML. After mastering the basics of shiny and I'm still not happy with the functions and layouts of the shiny. This is the time for us to learn more about web development toolkits, such as HTML/CSS/JavaScript. 

* __HTML__ is used to design the UI of the app and shouldn't be too difficult in Shiny, as most of the HTML tags are already incorporated into `tags` in Shiny.
* __CSS__ is used to format the HTML elements without harming the web-app. Of course, CSS can be igored, if you are already satisified with the format of the pages.
* __JavaScript__ is programming language that used to interact with the UI pages. JS can process complicated operations, such as plotting a sankey diagram in our web-app. There are a lot of other outstanding mature visualization JS packages (d3.js, echarts, google charts, etc) can be borrowed in Shiny. JavaScript is developing rapidly and recently Javscript is also capable of doing backend jobs (such as Node.js, Tensorflow.js).

Therefore JavaScript is a desired skill we should pick up if we would like to build some more beautful Shiny Apps.

## Integration of Shiny and JavaScript 

### Load JavaScript in Shiny

In HTML, we can use `<script>` tag to include the javascript. For example: 

* load remote js code: include d3.js version 5 using `<script src="https://d3js.org/d3.v5.min.js"></script>`
* load local js code:  include your sankey.js using `<script src="./js/sankey.js"></script>`

In Shiny, we can similarly:

* load remote js code: using `tags$script(src="https://d3js.org/d3.v5.min.js")`
* load local js code:  
    + using `tags$script(src="sankey.js")` when the js code is placed in `www` directory
    + using `includeScript('./js/sankey.js')` when the js code is place anywhere else

### From JavaScript to Shiny
```
Shiny.setInputValue(id, value);
```
### From Shiny to JavaScript

* send message in shiny: `session$sendCustomMessage(type, message)`. In thi s app, whenever the # bins changes, shiny session sends a message {"time": , "bins", } to JavaScript.
* recieve message in javascript: `Shiny.addCustomMessageHandler(type, function(message) {...});`. In this app, JS receives the

## Reference

* https://shiny.rstudio.com/articles/communicating-with-js.html