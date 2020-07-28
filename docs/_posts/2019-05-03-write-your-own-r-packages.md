---
title: "Write Your Own R Packages"
excerpt: "This post is to write my own util package to wrap all my udfs with a neat documentation."
date: 2019-05-03 14:47:00 +0800
categories:
  - notes
tags:
  - R
toc: true
toc_sticky: true
---

## Introduction

A set of user-defined functions (udf) or utility functions are helpful to simpify our code and avoid repeat the same typing for daily analysis work. Previously, I saved all my R functions to a single R file. Whenever I want to use them, I can simpily `source` the R file to import all functions. This is a simple but not perfect approach, especially when I want to check the documentation of certain functions. It was quite annonying that you can't just type `?func` to navigate to the help file. So, this is the main reason for me to write my own util package to wrap all my udfs with a neat documentation.

## Pre-requisite

In order to write your R pacagkes, you need pacakges of `devtools` and `roxygen2` for the minimal.

* devtools: for development actions, such as document, install, build and test
* roxygen2: for R package documentation

```{r}
# install the pre-requisite
install.packages(c('devtools','roxygen2'))
```

## Step1: Create A New R Project for Package

The first step is to create a new project in your rstuod by [File] > [New Proejct...]

![](https://github.com/6chaoran/data-story/raw/master/tutorial/write-r-package/image/01_new_project.JPG)

Choose a [New Project].

![](https://github.com/6chaoran/data-story/raw/master/tutorial/write-r-package/image/02_r_packages_using_devtools.JPG)

Choose [R Packages using devtools].

![](https://github.com/6chaoran/data-story/raw/master/tutorial/write-r-package/image/03_fill_in_package_name_path.JPG)

Fill in your package name and location. 

## Step2: Fill in DESCRIPTION

Once the projct is created, we should get files organized as following:

![](https://github.com/6chaoran/data-story/raw/master/tutorial/write-r-package/image/04_package_file_structure.JPG)

* R folder: is the place to put R functions
* man folder: is the manual/docment for functions
* DESCRIPTION: general information for this package
* NAMESPACE: exported functions for this pacakge

Let's fill in the DESCRIPTION file for general information for this package.

![](https://github.com/6chaoran/data-story/blob/raw/tutorial/write-r-package/image/05_DESCRIPTION.JPG)

## Step3: Write Your R Functions & Documents

Now this is core part of your R package. Define your first simple function, maybe for example quantiles for a dataframe.

```r
show_quantile <- function(df, q = seq(0,1,0.1)){
  res <- list()
  for(v in colnames(df)){
    if(is.numeric(df[,v])){
      res[[v]] <- quantile(df[,v],q)
    }
  }
  return(res)
}
```

The beautiful part comes here, you can use `'#` symbol to decorate the document for the functions.

* first line: title of the function
* third line: description of the function
* @param: document for the arguement
* @return: document for the return
* @export: indicator whether to export to NAMESPACE
* @examples: define the examples and test code
* @importFrom: import function from other package

more details is [here](https://cran.r-project.org/web/packages/roxygen2/vignettes/rd.html) on roxygen2.    

And the resulting R code looks like this:

```r
#'show_quantile (title)
#'
#' calculate the quantiles for all numeric columns (description)
#'
#' @param df dataframe that quantile function performed on (function arguement)
#' @param q quantiles, defaults to 0 to 1, with step of 0.1 (function arguement)
#' @return list (return)
#' @export (add this, if you want the functions availabe in NAMESPACE)
#' @examples (define the examples/test code)
#' data(iris)
#' show_quantile(iris)
show_quantile <- function(df, q = seq(0,1,0.1)){
  res <- list()
  for(v in colnames(df)){
    if(is.numeric(df[,v])){
      res[[v]] <- quantile(df[,v],q)
    }
  }
  return(res)
}
```

Document the package by typing `devtools::document()`

## Step4: Install and Test

Run `devtools::install()` to install the package and we can observe that the package is already available from rstudio under the packages tab.
![](https://github.com/6chaoran/data-story/raw/master/tutorial/write-r-package/image/07_install.JPG)

To test the package, we can restart the R session and use our favorite `library` command.

![](https://github.com/6chaoran/data-story/raw/master/tutorial/write-r-package/image/08_help_file.JPG)

When we type `?show_quantiles`, we can see the help information on the side panel.

## (Optional) Step5: Push package to Git

If you push the code to the git, we actually can install from cloud, so that sharing with others become easier.
I uploaded my code to Github, so other people can use `devtools` to install this package as well.

![](https://github.com/6chaoran/data-story/raw/master/tutorial/write-r-package/image/09_install_from_github.JPG)

The source code is available [here](https://github.com/6chaoran/data-story/tree/master/tutorial/write-r-package/myutils) on github

## Reference

* [Developing Packages with RStudio](https://support.rstudio.com/hc/en-us/articles/200486488-Developing-Packages-with-RStudio)
* [R Packages](http://r-pkgs.had.co.nz/)
* [Writing an R package from scratch](https://hilaryparker.com/2014/04/29/writing-an-r-package-from-scratch/)
