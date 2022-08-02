---
title: "Modified `readthedown` RMarkdown template for stylish analytical documents"
date: 2022-08-01 19:30:01 +0800
classes: wide
categories:
- visualization
tags:
- css
- rmarkdown
excerpt: "This is a modified `readthedown` rmarkdown template, which is greatly inspired and modified based on [juba/rmdformats](https://github.com/juba/rmdformats) package. `readthedown` offer a similar [sphnix](https://www.sphinx-doc.org/en/master/) style, which is commmonly used in various python package documentations. I personally very much like the `readthedown` style and hence dive a little bit on the source code to figure out ways to make it easier for further customization."
---

In [previous post](https://6chaoran.github.io/data-story/visualization/tips-of-drafting-r-markdown-document/), I shared how to effectively use various packages to create `htmlwidget` to build interactive analytical documents in `readthedown` template, which I enjoyed for the most of time. However I feel tedious that I need copy the same `yaml` configuration, `css` code, setup R chucks every time for new projects. That's the motivtaion of this [modified `readthedown` rmarkdown template](https://github.com/6chaoran/readthedown) is to ease the process of creating documents with the same style. It offers out-of-box template with predefined styles, configurations and is greatly inspired and modified based on [juba/rmdformats](https://github.com/juba/rmdformats) package. Now, all you need is just a few clicks on Rstudio menu to start a fresh template.

## What's New

1. __Included logos__   
created two placeholders for logos. `logo` is located at top-left corner in the sidebar, `logo2` is located at the top-right corner of the content. The `logo` and `logo2` are wrapped inside a `div` container with class `logo` and `logo2` respectively. This will be helpful when we need adjust the logo size in the `css` file.
2. __Included favicon__   
favicon is pretty small thing, but it's cool to share the html document that has your own icon instead of a "grey globe". The `favicon` option is made available in the yaml and can be easily replaced with your own favicon image.
3. __Improved author profile section__   
added a `avatar` option under `auther`, which can be easily defined at the rmarkdown yaml header.
4. __Colored block quote__    
to differentiate level of attention of the block quote, css classes of `info`, `warn` and `err` are created to show in different color.  
5. __Fixed css style for details & summary__   
`details` tag is useful html element to display or hide detail information. The expanding triagnle is somehow masked by the other setting in the origin template. A fix is applied to resolove it.
6. __Handy css classes for mulitple columns__   
predefined css classes to make it easier to show plots, tables in multiple columns layout side by side. It's more clear to check out the default template from the package.


## YAML configuration from `readthedown`

```yaml
title: "Readthedown Example"
date: "`r Sys.Date()`"
author: 
  - name: "Liu Chaoran <6chaoran@gmail.com>"
    avatar: ./logo/avatar.png
    title: Data Scientist
output:
  readthedown::readthedown:
    highlight: kate
    fig_width: 14
    fig_height: 8
    number_sections: true
    code_folding: none
    logo: ./logo/logo.png
    logo2: ./logo/logo2.png
    favicon: ./logo/favicon.png
    css: style.css
```

## Futher Customization

If the current template can't satisify you, you should try to modify the `style.css` file in the default `logo` directory. Some simple settings (e.g. background color, font size) can be easily changed with some basic knowledge of `css`. Alternatively, W3 School is good choices to copy some codes from.

```css
#content {
  max-width: 95%;
}

#sidebar > h2 {
  background-color: darkblue;
}

.logo {
  background-color: darkblue;
}

h1, h2, h3 {
  color: darkblue;
}
/* some other css settings ... */
```


## Rendered HTML document

The out-of-box RMarkDown template can be rendered to the following HTML document. Should you check out the [package](https://github.com/6chaoran/readthedown) and the [template](https://github.com/6chaoran/readthedown/blob/master/inst/rmarkdown/templates/readthedown/skeleton/skeleton.Rmd), if this is interesting to you.

<!-- <iframe src="https://htmlpreview.github.io/?https://raw.githubusercontent.com/6chaoran/data-story/master/r-markdown/readthedown-template.html" allowfullscreen = true width="100%" height="650" style="border:none;"></iframe> -->

<iframe src="https://6chaoran.github.io/data-story/assets/document/readthedown-template.html" allowfullscreen = true width="100%" height="650" style="border:none;"></iframe>

