---
title: "Released a DataFrame summarytool for Jupyter Notebook"
date: 2021-03-11 23:02:01 +0800
toc: true
toc_sticky: true
categories: 
  - visualization
tags:
  - python
excerpt: "Want to include a data summary as quick reference in your Jupyter notebooks ? I used to have `summarytools` package in R to do this. I miss that one when I'm doing python projects. So I developed a similar python function with some additional widgets. Please check out this post if you are interested."
---

## About the package

This is python version of `summarytools`, which is used to generate standardized and comprehensive summary of pandas `DataFrame` in Jupyter Notebooks.

The idea is originated from the [`summarytools`](https://github.com/dcomtois/summarytools) R package . Only dfSummary function is made available for now. I also added two html widgets (_collapsible/tabbed view_) to avoid displaying lengthy content.

## Quick Start

### default view

out-of-box `dfSummary` function will generate a HTML based data frame summary.

```python
import pandas as pd
from summarytools import dfSummary
titanic = pd.read_csv('./data/titanic.csv')
dfSummary(titanic)
```
![](https://github.com/6chaoran/jupyter-summarytools/raw/master/images/dfSummary.png)

If too many data summaries are included in the same notebook, the following two widgets should be able to help.

### collapsible view

```python
import pandas as pd
from summarytools import dfSummary
titanic = pd.read_csv('./data/titanic.csv')
dfSummary(titanic, is_collapsible = True)
```
![](https://github.com/6chaoran/jupyter-summarytools/raw/master/images/collapsible.gif)

### tabbed view

```python
import pandas as pd
from summarytools import dfSummary, tabset
titanic = pd.read_csv('./data/titanic.csv')
vaccine = pd.read_csv('./data/country_vaccinations.csv')
vaccine['date'] = pd.to_datetime(vaccine['date'])

tabset({
    'titanic': dfSummary(titanic).render(),
    'vaccine': dfSummary(vaccine).render()})
```
![](https://github.com/6chaoran/jupyter-summarytools/raw/master/images/tabbed.gif)

## Export as HTML

when export jupyter notebook to HTML, make sure `Export Embedded HTML` extension is installed and enabled.
![](https://github.com/6chaoran/jupyter-summarytools/raw/master/images/embedded_html.png)

Using the following bash command to retain the data frame summary in exported HTML.

```bash
jupyter nbconvert --to html_embed path/of/your/notebook.ipynb
```

## Installation

detail is available at [https://github.com/6chaoran/jupyter-summarytools](https://github.com/6chaoran/jupyter-summarytools)
