---
layout: single
title: "Featured Projects"
permalink: /projects/
author_profile: true
classes: wide

summarytools:
  - image_path: assets/images/projects/summarytools-header.png    
    alt: "placeholder image 3"
    title: "DataFrame Summary Tool in Jupyter Notebook"
    excerpt: 'Inspired by R summarytools package, I replicated a similar package in Jupyter Notebook. 
    
    * Check [https://pypi.org/project/summarytools](https://pypi.org/project/summarytools) for installation & quick start.

    * Check [https://github.com/6chaoran/jupyter-summarytools](https://github.com/6chaoran/jupyter-summarytools) for the source code.
    
    
    Click `Read More` to continue with the post.'

    url: "/visualization/summarytools-for-jupyter-notebook/"
    btn_label: "Read More"
    btn_class: "btn--primary"

face2bmi:
  - image_path: assets/images/projects/face2bmi-header.png    
    alt: "placeholder image 2"
    title: "Face Recognition & BMI Prediction using Keras"
    excerpt: 'In this post, we build a model that provides end-to-end capability of detecting faces from image and predicting the BMI, Age and Gender for each detected persons.    '
    url: "/deep-learning/detect-faces-and-predict-BMI-using-keras/"
    btn_label: "Read More"
    btn_class: "btn--primary"

deepfm:
  - image_path: https://www.researchgate.net/profile/Huifeng_Guo/publication/318829508/figure/fig1/AS:522607722467328@1501610798143/Wide-deep-architecture-of-DeepFM-The-wide-and-deep-component-share-the-same-input-raw.png  
    alt: "placeholder image 1"
    title: "Implement DeepFM model using Keras"
    excerpt: '
    
Wide and deep architect has been proven as one of deep learning applications combining memorization and generalization in areas such as search and recommendation. Google released its wide&deep learning in 2016.


* wide part: helps to memorize the past behavior for specific choice

* deep part: embed into low dimension, help to discover new user, product combinations


Later, on top of wide & deep learning, deepfm was developed combining DNN model and Factorization machines, to further address the interactions among the features.'
    url: "/deep-learning/recsys/deepfm-for-recommendation/"
    btn_label: "Read More"
    btn_class: "btn--primary"
---
{% include feature_row id="summarytools" type="left" %}
{% include feature_row id="face2bmi" type="left" %}
{% include feature_row id="deepfm" type="left" %}

<style>
  #page-title {
    margin: 20px 0 40px 0;
  }
.page__content .archive__item-title {
  margin-top: 0;
}
</style>