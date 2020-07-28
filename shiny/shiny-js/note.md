* from Javascript to Shiny
```
Shiny.setInputValue(id, value);
```
* from Shiny to Javascript
    + send message in shiny: `session$sendCustomMessage(type, message)`. In thi s app, whenever the # bins changes, shiny session sends a message {"time": , "bins", } to JavaScript.
    + recieve message in javascript: `Shiny.addCustomMessageHandler(type, function(message) {...});`. In this app, JS receives the json message and create a p-tag element and append it to the history section. It also makes the text red if # bins > 50.

#### Reference
* https://shiny.rstudio.com/articles/communicating-with-js.html