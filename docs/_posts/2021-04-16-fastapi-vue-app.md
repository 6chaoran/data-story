---
title: "Build an API App backed by FastAPI and Vue.js"
date: 2021-04-16 10:02:01 +0800
toc: true
toc_sticky: true
categories: 
  - visualization
  - data-engineering
tags:
  - python
header:
  image: /assets/images/posts/fastapi-vue-banner.png
excerpt: "Presenting an API is never going to be attractive. In this post, I documented my approach of developing a web page on top of existing API using FastAPI + Vue.js technology stack."
---
API is usually considered as the last step of a Machine Learning project deliverable.
But when it comes to the demonstration, it's not attractive to ask audience to look at the Swagger/Postman API testing screen during your presentation. In a well-structured data science team, you may have the luxury of support from front-end engineers to build a web page for your prototype. But in most scenarios, data scientists are working on their own to complete the minimal variable data product. 

There are different alternatives (such as R Shiny, Plotly Dash, Flask, etc), to achieve the purpose. In this post, I'm going to document my approach of developing a web page on top of existing API using FastAPI + Vue.js.

The resulting API App is demonstrated in the screenshot below:

<img src="https://raw.githubusercontent.com/6chaoran/fastapi-vue-app/master/images/screenshot2.gif" style="width:100%;">

# Prerequisite

## FastAPI
FastAPI ([Document](https://fastapi.tiangolo.com/) &#124; [GitHub](https://github.com/tiangolo/fastapi)) is my personal preference of API framework because of its high performance and comprehensive documentation.

In this post, FastAPI is used to wrap the ML model to a working API and host a static HTML on the same port. By doing this, we just need to launch a single service to make both API and Web-App working.

## Vue.js

Vue.js ([Document](https://vuejs.org/v2/guide/) &#124; [GitHub](https://github.com/vuejs/vue)) is one of the popular Javascript frameworks. I'm personally more appealing to Vue.js as I feel it's easier to pick up for quick prototypes, though React.js or Angular.js are more widely used.

Vue.js provides a series of plugins to make life easier:

+ [__Vue2__](https://vuejs.org/v2/guide/): the core Vue.js library. version 2 is used, as some legacy packages are still incompatible with the ongoing version3 of Vue.
+ [__Vuetify__](https://vuetifyjs.com/en/): the material design framework, which provides quick and beautiful UI components for Vue. Vue3 compatible version is still developing, and that's the main reason that I still keep it on Vue2.
+ [__Vue Router__](https://router.vuejs.org/): router for Vue.js, which provides the capability of multi-page web app. 

Some other packages are not in Vue suite, but also useful to have:

+ [__axios__](https://github.com/axios/axios): the popular Javascript library that works with API.
+ [__vue-echarts__](https://github.com/ecomfe/vue-echarts): interactive charting library that works with Vue2 & Vue3.

# Development Cycle

Having a brief understanding of the FastAPI and Vue.js, we are ready to start our development cycle of API APP, which generally consists of the following steps.

1. model development → output ETL pipeline, model object
2. API wrapping → output an API, API spec (swagger/openAI)
3. App design → output a web app
4. [optional] containerization → output a docker image

As this post is primarily focusing on API App development, the process of model building and API wrapping will be only briefly described.

If you are already familiar with model developing and API wrapping, you can skip to the [__App Design__](#app-design) section directly.  

# Model Development

## Data Preparation

My favorite dataset - [_Titanic_](https://raw.githubusercontent.com/6chaoran/fastapi-vue-app/master/data/titanic.csv), is used in this post for demonstration. The data preparation process can be described as following:

1. load csv data using pandas
2. split data into training and validation set
3. encode the categorical variables so that ML model can understand

Don't worry about `encode_cat_variables` function, if you intend to run this code snippet. You can refer to the completed code in my [GitHub repository](https://github.com/6chaoran/fastapi-vue-app).

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle
from utils import encode_cat_variables

# define variables
fnames_cat = ['Pclass', 'Sex', 'Embarked']
fnames_num = ['Age', 'Fare', 'SibSp', 'Parch']
fnames = fnames_cat + fnames_num

# load data
data = pd.read_csv('./data/titanic.csv')

# split train / valid
train, valid = train_test_split(data, stratify=data.Survived, train_size=0.7)

# encode categorical variables
train, le = encode_cat_variables(train, fnames_cat)
valid, le = encode_cat_variables(valid, fnames_cat, le)
```

## Model Building

LightGBM model is used in this post. LightGBM has a unique function of keeping categorical features, so we don't have to do the one-hot encoding. As a result of model building, we saved the trained model object and variable encoders to prepare for API wrapping.

```python
# convert to lightgbm Dataset
dtrain = lgb.Dataset(train[fnames].values, train.Survived)
dvalid = lgb.Dataset(valid[fnames].values, valid.Survived)

params = {
    'objective': 'binary',
    'eta': 0.1,
    'metric': 'auc'
}

model = lgb.train(params,
                  dtrain,
                  num_boost_round=100,
                  valid_sets=[dtrain, dvalid],
                  valid_names=['train', 'valid'],
                  feature_name=fnames,
                  categorical_feature=fnames_cat,
                  verbose_eval=10)

# save model & encoder
model.save_model(filename = "./saved_model/model.txt")
with open('./saved_model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
```

# API Wrapping

## Load saved model

All the saved model and related utility functions need to be restored in API. We can load them in the beginning of API script.

```python
import pickle
import lightgbm as lgb

# load encoder
with open('./saved_model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
# load model
model = lgb.Booster(model_file='./saved_model/model.txt')
```


## Define payload data model

Payload (API input) data model need to be defined in a custom python class `TitanicFeature`.

The data model is useful to

* constrain the inputs for validation
* convert inputs to desired types if possible

```python
from pydantic import BaseModel, Field

# data model of predictors
class TitanicFeature(BaseModel):
    Age: int = Field(..., example=20)
    Pclass: int = Field(..., example=1)
    Sex: str = Field(..., example='male')
    SibSp: int = Field(..., example=1)
    Parch: int = Field(..., example=1)
    Fare: float = Field(..., example=120)
    Embarked: str = Field(..., example='S')
```

## Define API methods

A POST method needs to be defined to allow the users to post their input to the ML model, and the prediction score and [SHAP](https://github.com/slundberg/shap) values will be returned.

```python
@app.post("/predict")
async def predict(payload: TitanicFeature):
    # convert the payload to pandas DataFrame
    input_df = pd.DataFrame([payload.dict()])
    # encoded all the categorical variables
    input_df_encoded, _ = encode_cat_variables(input_df, list(le.keys()), le)
    # output the prediction score
    score = model.predict(input_df_encoded)[0]
    # output the SHAP values
    shap_values = model.predict(input_df_encoded, pred_contrib=True)[0]
    # remove the last term - bias
    shap_values = shap_values[:-1]
    # desc sort SHAP variables by absolute value
    shap_values = shap_values[np.argsort(-np.abs(shap_values))]
    shap_values = [
        {"name": fnames[i], "value": np.round(v, 4)} for i, v in enumerate(shap_values)
    ]
    return {
        'score': score,
        'shap_values': shap_values
    }
```

## Define APP entry HTML

A GET method needs to be defined to get the HTML entry point (`app.html`), which is the placeholder for our Vue application.

```python
from fastapi.responses import FileResponse

@app.get("/app")
def read_index():
    return FileResponse("./app.html")
```

## Launch and test API

Lastly, we should run the following command to launch the API for testing.

* __OpenAPI__: go to http://localhost:8005/docs to test out if API methods return the expected results.
* __WebAPP__: go to http://localhost:8005/app to view the hosted web page. For now, the web page should be blank, as we haven't put anything inside the `app.html`.

```bash
# launch API (from api.py) on port 8005
uvicorn api:app --reload --port 8005
```

# App Design

There are generally two ways to develop the Vue.js Applications.

1. In browser development, and use `CDN` version of Javascript packages
2. In Command Line Interface (CLI) development, and use `npm` version of Javascript packages

As development in browser is much easier to pick up and suitable for smaller and simpler web page, the `CDN` approach will be used in this post. However when the scale of web app goes larger, CLI approach is strongly recommended.

Vue has its own [CLI tool](https://cli.vuejs.org/guide/) and the readers can follow it's official website to learn the basics of `Vue CLI`.

Visual Studio Code is very recommended to develop the web application. As the `Emmet` is supported in Visual Studio Code, you can just type `doc`, after you create the blank `app.html` file. Hit `Enter` button and the HTML template will be autocompleted.

## Include Javascript Libraries

Just like start any R / Python scripts, we need to load the required libraries. Javascript libraries can be loaded in HTML, and usually placed at bottom of the `body` tag, so that the page loading won't be largely affected.

In the following code snippet, we loaded three packages (`Vue2`, `Vuetify2`, `axiso`). In case, you need to add any other libraries, just look for the `CDN section` of the package installation guide and copy the `script` tag into your HTML.

```html
<body>
    <script src="https://cdn.jsdelivr.net/npm/vue@2.x/dist/vue.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/vuetify@2.x/dist/vuetify.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
</body>
```

## Create Vue Instance 

The Vue instance is also defined in `<script>` tag. 

* `data` section: defines the data two-way binding to the Vue application, which means the variables in `data` can used to render in UI or updated by UI.
* `methods` section: defines the function used in Vue app.

```html
    <script>
        new Vue({
            el: '#app',
            vuetify: new Vuetify(),
            data: function(){
                return {}
            },
            methods: {},
        })
    </script>
```

## Create Vuetify-theme App

The power of `vuetify` helps us quickly build a beautify theme. All the `vuetify` tags are started with `v`. We just need copy the following `<div>` chunk into the `<body>` of HTML, to create the UI of web app.

* `<v-app>`: define an application with `vuetify` theme
* `<v-app-bar>`: create an application bar
* `<v-main>`: create a placeholder for main content

More UI components can be explored on the `Vuetify` website.

```html
<div id="app">
    <v-app>
        <v-app-bar app dense dark color="purple">
            <v-app-bar-title>Titanic Demo</v-app-bar-title>
        </v-app-bar>
        <v-main></v-main>
    </v-app>
</div>
```

The resulting web app should look like the screenshot below.

![](https://raw.githubusercontent.com/6chaoran/fastapi-vue-app/master/images/app-design-step1.png)

## Create model input form and submit button

After the theme and layout is set, we will create the input panel to allow users to type in their parameters for model prediction. Copy the `<v-navigation-drawer>` into the `<v-main>` to create a side panel.

* `<v-navigation-drawer>`: creates a navigation panel, which can be fixed or temporary.
* `class="my-3"`: `m` means margin, `y` means vertical, `3` is the spacing (3 x 4px = 12px). Similarly, you will guess what does `my-6` mean. More can be found [here](https://vuetifyjs.com/en/styles/spacing/#how-it-works)
* `<v-divider>`: creates a styled horizontal line.
* `<v-text-field>`: creates a text input. For simplicity, I used text inputs for all variables, but select type input should be used for categorical variables, such as `Sex` or `Embarked`. 
* `v-for`: is for-loop syntax in Vue. It will loop over the `payload` object and create a `<v-text-field>` input for each predictor.
* `v-model`: defines the two-way binding between Vue and UI, so when the app is launched, the UI will take the `name` variable in `payload` for the default value, and whenever the user updates the UI, the `name` value in `payload` will be updated as well.
* `:label`: is short-hand of `v-on:label`, which tells Vue that `label` will take a variable instead of a string as argument. If we use `label="p.name"`, then each input field in the UI will have the same label of `p.name`, which is not expected here.
* `<v-btn>`: creates a styled button. `@click` is same as `onclick` in native Javascript. `@click=call_api` means `call_api()` function will be called if the button is clicked.


```html
<v-navigation-drawer right absolute>
    <v-container>
        <div class="my-3">Side Panel</div>
        <v-divider></v-divider>
        <div class="my-6">
            <v-text-field 
                v-for="p in payload" 
                v-model="p.value" 
                :label="p.name" 
                :key="p.name"
                dense
                outlined>
            </v-text-field>
        </div>
        <div class="my-3">
            <v-btn 
                text 
                color="purple" 
                @click="call_api">
                predict
            </v-btn>
        </div>
    </v-container>
</v-navigation-drawer>
```

We defined the `payload` in `data` section to store and update all the predictors for the model.

```js
data: function () {
    return {
        payload: [
            { name: "Age", value: 20, type: "int" },
            { name: "Pclass", value: "1", type: "str", items: ["1", "2", "3"] },
            { name: "Sex", value: "male", type: "str", items: ["male", "female"] },
            { name: "SibSp", value: 1, type: "int" },
            { name: "Parch", value: 1, type: "int" },
            { name: "Fare", value: 120, type: "int" },
            { name: "Embarked", value: "S", type: "str", items: ["S", "C", "Q"] },],
        score: null
    }
},
```

The resulting web app should look like the screenshot below.

![](https://raw.githubusercontent.com/6chaoran/fastapi-vue-app/master/images/app-design-step2.png)

## Call API from Vue

When the App received inputs (`payload`) from users, we are ready to make the API to get the prediction results. Remember we created a button to call `call_api` function ? We will define it here in the `methods` section of Vue instance.

* `this`: refers to Vue instance. `this.payload` points to the `payload` object we created and updated by users

```js
methods: {
    call_api: function () {
        // reformat the payload from [{name: Age, value: 10}, ...] to {Age:10, ...}
        payload = this.payload.reduce((acc, cur) => ({ ...acc, [cur.name]: cur.value }), {})
        axios.post("/predict", this.payload)
            .then(resp => resp.data)
            .then(data => {
                this.score = data.score
            })
            .catch(e => console.log(e))
    },
},
```

## Display the model output

If the API call was tested okay, we should plan how to display the result on UI. Copy the following `<v-container>` chunk inside `<v-main>`, but outside `<v-navigation-drawer>` to display the payload, model prediction score.

* `<v-container>`: creates a `<div>` but with pre-defined spacings.
* {% raw  %}`{{ }}`{% endraw %}: is the placeholder to display string in UI.
* `v-if`: conditional display. The `<h1>` content will only be shown when score is not null.

```html
<v-container id="main">
    <h1>Payload</h1>
    {% raw  %}{{ payload }}{% endraw %}
    <h1 v-if="score">Model Score</h1>
    {% raw  %}{{ score }}{% endraw %}
</v-container>
```
The resulting web app should look like the screenshot below.

![](https://raw.githubusercontent.com/6chaoran/fastapi-vue-app/master/images/app-design-step3.png)

Play around with all the buttons in the web page to make sure the behavior is working in the expected way. 

Please refer to 

* [`index.html`](https://github.com/6chaoran/fastapi-vue-app/blob/master/index.html) for a minimal Vue.js application
* [`app.html`](https://github.com/6chaoran/fastapi-vue-app/blob/master/app.html) for a complete Vue.js application

Congratulations , you now have a Web App serving at `http://localhost:8005/app` together with your working API.