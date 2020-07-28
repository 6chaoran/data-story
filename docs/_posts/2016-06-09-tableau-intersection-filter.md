---
title: "Tableau Intersection Filter Tutorial"
date: 2016-06-09 16:16:01 +0800
categories: 
  - visualization
tags:
  - tableau
classes: wide
---

If you used Tableau before, you will know that the filters in Tableau are union/or selection.Let’s take the table below for example.   

If you are going to create a filter and select product a & b, tableau will show client A,B,C and E instead of A,C. It’s because the filters will show us the list of clients who purchased product a or b, instead of product a and b.

![image](https://6chaoran.files.wordpress.com/2016/06/intersection-filter-data-table.png?w=700)
![image](https://6chaoran.files.wordpress.com/2016/06/capture1.png?w=700)

### the idea

Firstly, create a variable to count the selection of products. Then create another variable to count the selection of products from each client. If these two variable equal, it means that the clients purchased all products that we picked.

### steps to construct intersection filter

* Step 1: 
    * Create Calculation Field __[# Product]__
    * _TOTAL(COUNTD([Product]))_
    * _Compute using: Table Down_
* Step 2:    
    * Create Calculation Field [# Product Selected]
    * _TOTAL(COUNTD([Product]))_
    * _Compute using: Client_
* Step 3:
    * Duplicate __[Client]__
    * place __[Client (copy)]__ to Marks as Dimension
* Step 4:
    * Create Calculation Field __[intersection filter]__
    * __[# Product]__ = __[# Product Selected]__
* Step 5:
    * Place __[intersection filter]__ to Filters panel and select True.

![image](https://6chaoran.files.wordpress.com/2016/06/capture2.png?w=700)

### put sheets into dashboard

Click [here](https://public.tableau.com/views/IntersectionFilter/intersectionfilter?:embed=y&:display_count=yes&:showTabs=y) to view the Tableau example.

![image](https://6chaoran.files.wordpress.com/2016/06/capture3.png?w=700)