# Revisit Titanic Data Using Apache Spark
This post is mainly to demonstrate the pyspark API (Spark 1.6.1), using Titanic dataset, which can be found here (train.csv, test.csv). 

##Content

1. Data Loading and Parsing
2. Data Manipulation
3. Feature Engineering
4. Apply Spark ml/mllib models

### 1. Data Loading and Parsing

#### a. Data Loading

sc is the SparkContext launched together with pyspark. Using sc.textFile, we can read csv file as text in RDD data format and data is separated by comma.

```python
train_path='/Users/chaoranliu/Desktop/github/kaggle/titanic/train.csv'
test_path='/Users/chaoranliu/Desktop/github/kaggle/titanic/test.csv'
# Load csv file as RDD
train_rdd = sc.textFile(train_path)
test_rdd = sc.textFile(test_path)
```

Let’s look at the first 3 rows of the data in RDD.    
```python
train_rdd.take(3)
```
```
  [u'PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked',
  u'1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S',
  u'2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C']
```

#### b. Parse RDD to DataFrame

Inspired from R DataFrame and Python pandas, Spark DataFrame is the newer data format supported by Spark. We are going to transform RDD to DataFrame for later data manipulation.

steps to transform RDD to DataFrame   

step1: remove header from data   
step2: separate each row by comma and convert to tuple   
step3: names each column by the header   

```python
# Parse RDD to DF
def parseTrain(rdd):
 
    # extract data header (first row)
    header = rdd.first()
    # remove header
    body = rdd.filter(lambda r: r!=header)
def parseRow(row):
        # a function to parse each text row into
        # data format
        # remove double quote, split the text row by comma
        row_list = row.replace('&quot;','').split(&quot;,&quot;)
        # convert python list to tuple, which is
        # compatible with pyspark data structure
        row_tuple = tuple(row_list)
        return row_tuple
 
    rdd_parsed = body.map(parseRow)
 
    colnames = header.split(&quot;,&quot;)
    colnames.insert(3,'FirstName')
 
    return rdd_parsed.toDF(colnames)
 
def parseTest(rdd):
    header = rdd.first()
    body = rdd.filter(lambda r: r!=header)
 
    def parseRow(row):
        row_list = row.replace('&quot;','').split(&quot;,&quot;)
        row_tuple = tuple(row_list)
        return row_tuple
 
    rdd_parsed = body.map(parseRow)
 
    colnames = header.split(&quot;,&quot;)
    colnames.insert(2,'FirstName')
 
    return rdd_parsed.toDF(colnames)
 
train_df = parseTrain(train_rdd)
test_df = parseTest(test_rdd)
```

Now let’s take a look at the DataFrame

```python
train_df.show(3)
```
+-----------+--------+------+---------+--------------------+------+---+-----+-----+----------------+-------+-----+--------+
|PassengerId|Survived|Pclass|FirstName|                Name|   Sex|Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|
+-----------+--------+------+---------+--------------------+------+---+-----+-----+----------------+-------+-----+--------+
|          1|       0|     3|   Braund|     Mr. Owen Harris|  male| 22|    1|    0|       A/5 21171|   7.25|     |       S|
|          2|       1|     1|  Cumings| Mrs. John Bradle...|female| 38|    1|    0|        PC 17599|71.2833|  C85|       C|
|          3|       1|     3|Heikkinen|         Miss. Laina|female| 26|    0|    0|STON/O2. 3101282|  7.925|     |       S|
+-----------+--------+------+---------+--------------------+------+---+-----+-----+----------------+-------+-----+--------+

#### c.Combine Train/Test Data

```python
## Add Survived column to test
from pyspark.sql.functions import lit, col
train_df = train_df.withColumn('Mark',lit('train'))
test_df = (test_df.withColumn('Survived',lit(0))
                  .withColumn('Mark',lit('test')))
test_df = test_df[train_df.columns]
## Append Test data to Train data
df = train_df.unionAll(test_df)
```

### 2.Data Cleaning/Manipulation
#### a.Convert Age, SibSp, Parch, Fare to Numeric

```python
df = (df.withColumn('Age',df['Age'].cast(&quot;double&quot;))
            .withColumn('SibSp',df['SibSp'].cast(&quot;double&quot;))
            .withColumn('Parch',df['Parch'].cast(&quot;double&quot;))
            .withColumn('Fare',df['Fare'].cast(&quot;double&quot;))
            .withColumn('Survived',df['Survived'].cast(&quot;double&quot;))
            )
df.printSchema()
```
```
root
 |-- PassengerId: string (nullable = true)
 |-- Survived: double (nullable = true)
 |-- Pclass: string (nullable = true)
 |-- FirstName: string (nullable = true)
 |-- Name: string (nullable = true)
 |-- Sex: string (nullable = true)
 |-- Age: double (nullable = true)
 |-- SibSp: double (nullable = true)
 |-- Parch: double (nullable = true)
 |-- Ticket: string (nullable = true)
 |-- Fare: double (nullable = true)
 |-- Cabin: string (nullable = true)
 |-- Embarked: string (nullable = true)
 |-- Mark: string (nullable = false)
```
#### b. Impute missing Age and Fare with the Average
```python
numVars = ['Survived','Age','SibSp','Parch','Fare']
def countNull(df,var):
    return df.where(df[var].isNull()).count()
 
missing = {var: countNull(df,var) for var in numVars}
age_mean = df.groupBy().mean('Age').first()[0]
fare_mean = df.groupBy().mean('Fare').first()[0]
df = df.na.fill({'Age':age_mean,'Fare':fare_mean})
```
missing Age, Fare are filled with the average value.   
```
{'Age': 263, 'Fare': 1, 'Parch': 0, 'SibSp': 0, 'Survived': 0}
```

### 3.Feature Engineering

#### a. Extract Title from Name

The idea is to create user-defined-function (udf) map on column “Name”.

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
 
## created user defined function to extract title
getTitle = udf(lambda name: name.split('.')[0].strip(),StringType())
df = df.withColumn('Title', getTitle(df['Name']))
 
df.select('Name','Title').show(3)
```

```
+--------------------+-----+
|                Name|Title|
+--------------------+-----+
|     Mr. Owen Harris|   Mr|
| Mrs. John Bradle...|  Mrs|
|         Miss. Laina| Miss|
+--------------------+-----+
only showing top 3 rows
```
#### b. Index categorical variable

Categorical feature is normally converted to numeric variable before apply machine learning algorithms. Here I just simply index the categories, which may not be the best for the conversion, as an unexpected correlation may be introduced in this method.

```python
catVars = ['Pclass','Sex','Embarked','Title']
 
## index Sex variable
from pyspark.ml.feature import StringIndexer
si = StringIndexer(inputCol = 'Sex', outputCol = 'Sex_indexed')
df_indexed = si.fit(df).transform(df).drop('Sex').withColumnRenamed('Sex_indexed','Sex')
 
## make use of pipeline to index all categorical variables
def indexer(df,col):
    si = StringIndexer(inputCol = col, outputCol = col+'_indexed').fit(df)
    return si
 
indexers = [indexer(df,col) for col in catVars]
 
from pyspark.ml import Pipeline
pipeline = Pipeline(stages = indexers)
df_indexed = pipeline.fit(df).transform(df)
 
df_indexed.select('Embarked','Embarked_indexed').show(3)
```

The categorical features are indexed in resulting data. Embarked is mapped S=>0, C=>1, Q=>2.

```
+--------+----------------+
|Embarked|Embarked_indexed|
+--------+----------------+
|       S|             0.0|
|       C|             1.0|
|       S|             0.0|
+--------+----------------+
only showing top 3 rows
```
#### c. Convert to label/features format

In order to apply ML/MLLIB, we need covert features to Vectors (either SparseVector or DenseVector).
```python
catVarsIndexed = [i+'_indexed' for i in catVars]
featuresCol = numVars+catVarsIndexed
featuresCol.remove('Survived')
labelCol = ['Mark','Survived']
 
from pyspark.sql import Row
from pyspark.mllib.linalg import DenseVector
row = Row('mark','label','features')
 
df_indexed = df_indexed[labelCol+featuresCol]
# 0-mark, 1-label, 2-features
# map features to DenseVector
lf = df_indexed.map(lambda r: (row(r[0], r[1],DenseVector(r[2:])))).toDF()
# index label
# convert numeric label to categorical, which is required by
# decisionTree and randomForest
lf = StringIndexer(inputCol = 'label',outputCol='index').fit(lf).transform(lf)
 
lf.show(3)
```
```
+-----+-----+--------------------+-----+
| mark|label|            features|index|
+-----+-----+--------------------+-----+
|train|  0.0|[22.0,1.0,0.0,7.2...|  0.0|
|train|  1.0|[38.0,1.0,0.0,71....|  1.0|
|train|  1.0|[26.0,0.0,0.0,7.9...|  1.0|
+-----+-----+--------------------+-----+
only showing top 3 rows
```
#### c. split back train/test data

```python
train = lf.where(lf.mark =='train')
test = lf.where(lf.mark =='test')
 
# random split further to get train/validate
train,validate = train.randomSplit([0.7,0.3],seed =121)
 
print 'Train Data Number of Row: '+ str(train.count())
print 'Validate Data Number of Row: '+ str(validate.count())
print 'Test Data Number of Row: '+ str(test.count())
```
```
Train Data Number of Row: 636
Validate Data Number of Row: 255
Test Data Number of Row: 418
```

## 4. Apply Models from ML/MLLIB
ML is built based on DataFrame, while mllib is based on RDD.   
I’m going to fit the logistic, decision tree and random forest models from ML packages.   
### Logistic Regression
```python
from pyspark.ml.classification import LogisticRegression
 
# regPara: lasso regularisation parameter (L1)
lr = LogisticRegression(maxIter = 100, regParam = 0.05, labelCol='index').fit(train)
 
# Evaluate model based on auc ROC(default for binary classification)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
 
def testModel(model, validate = validate):
    pred = model.transform(validate)
    evaluator = BinaryClassificationEvaluator(labelCol = 'index')
    return evaluator.evaluate(prod)
 
print 'AUC ROC of Logistic Regression model is: '+str(testModel(lr))
```
AUC ROC of Logistic Regression model is: 0.836952368823   
Logistic Regression model has a ROC 0.837 and we will compare the score with decision tree and random Forest.   

### Decision Tree and Random Forest

```python
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
 
dt = DecisionTreeClassifier(maxDepth = 3, labelCol ='index').fit(train)
rf = RandomForestClassifier(numTrees = 100, labelCol = 'index').fit(train)
 
models = {'LogisticRegression':lr,
          'DecistionTree':dt,
          'RandomForest':rf}
 
modelPerf = {k:testModel(v) for k,v in models.iteritems()}
 
print modelPerf
```
```
{'DecistionTree': 0.7700267447784003,
 'LogisticRegression': 0.8369523688232298,
 'RandomForest': 0.8597809475292919}
```
Without model tuning, random forest looks good for the prediction.   

The full python code can be found here
