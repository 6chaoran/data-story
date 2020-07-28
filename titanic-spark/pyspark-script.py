train_path='/Users/chaoranliu/Desktop/github/kaggle/titanic/train.csv'
test_path='/Users/chaoranliu/Desktop/github/kaggle/titanic/test.csv'

# Load csv file as RDD
train_rdd = sc.textFile(train_path)
test_rdd = sc.textFile(test_path)


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
		row_list = row.replace('"','').split(",")
		# convert python list to tuple, which is 
		# compatible with pyspark data structure
		row_tuple = tuple(row_list)
		return row_tuple

	rdd_parsed = body.map(parseRow)

	colnames = header.split(",")
	colnames.insert(3,'FirstName')

	return rdd_parsed.toDF(colnames)

def parseTest(rdd):
	header = rdd.first()
	body = rdd.filter(lambda r: r!=header)

	def parseRow(row):
		row_list = row.replace('"','').split(",")
		row_tuple = tuple(row_list)
		return row_tuple

	rdd_parsed = body.map(parseRow)

	colnames = header.split(",")
	colnames.insert(2,'FirstName')

	return rdd_parsed.toDF(colnames)

train_df = parseTrain(train_rdd)
test_df = parseTest(test_rdd)


## Add Survived column to test
## And append train/test data
from pyspark.sql.functions import lit, col
train_df = train_df.withColumn('Mark',lit('train'))
test_df = (test_df.withColumn('Survived',lit(0))
				  .withColumn('Mark',lit('test')))

test_df = test_df[train_df.columns]
df = train_df.unionAll(test_df)

## Data Cleaning/Manipulation
## Convert Age, SibSp, Parch, Fare to Numeric
df = (df.withColumn('Age',df['Age'].cast("double"))
			.withColumn('SibSp',df['SibSp'].cast("double"))
			.withColumn('Parch',df['Parch'].cast("double"))
			.withColumn('Fare',df['Fare'].cast("double"))
			.withColumn('Survived',df['Survived'].cast("double"))
			)

df.printSchema()

## Impute missing Age and Fare
numVars = ['Survived','Age','SibSp','Parch','Fare']
def countNull(df,var):
	return df.where(df[var].isNull()).count()

missing = {var: countNull(df,var) for var in numVars}
age_mean = df.groupBy().mean('Age').first()[0]
fare_mean = df.groupBy().mean('Fare').first()[0]
df = df.na.fill({'Age':age_mean,'Fare':fare_mean})


# Feature Enginnering
## 1. Extract Title from Name
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

## created user defined function to extract title
getTitle = udf(lambda name: name.split('.')[0].strip(),StringType())
df = df.withColumn('Title', getTitle(df['Name']))

## 2. Index categorical variable
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

## 3. Convert to label/features format
catVarsIndexed = [i+'_indexed' for i in catVars]
featuresCol = numVars+catVarsIndexed
featuresCol.remove('Survived')
labelCol = ['Mark','Survived']

from pyspark.sql import Row
from pyspark.mllib.linalg import DenseVector
row = Row('mark','label','features')

df_indexed = df_indexed[labelCol+featuresCol]
# 0-mark, 1-label, 2-features
lf = df_indexed.map(lambda r: (row(r[0], r[1],DenseVector(r[2:])))).toDF()
# index label 
lf = StringIndexer(inputCol = 'label',outputCol='index').fit(lf).transform(lf)

# split back train/test data
train = lf.where(lf.mark =='train')
test = lf.where(lf.mark =='test')

# random split further to get train/validate
train,validate = train.randomSplit([0.7,0.3],seed =121)

print 'Train Data Number of Row: '+ str(train.count())
print 'Validate Data Number of Row: '+ str(validate.count())
print 'Test Data Number of Row: '+ str(test.count())

# Apply Logsitic Regression
from pyspark.ml.classification import LogisticRegression

# regPara: regualrization parameter
lr = LogisticRegression(maxIter = 100, regParam = 0.05, labelCol='index').fit(train)

# Evaluate model based on auc ROC(default for binary classification)
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def testModel(model, validate = validate):
	pred = model.transform(validate)
	evaluator = BinaryClassificationEvaluator(labelCol = 'index')
	return evaluator.evaluate(pred)

from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier


dt = DecisionTreeClassifier(maxDepth = 3, labelCol ='index').fit(train)
rf = RandomForestClassifier(numTrees = 100, labelCol = 'index').fit(train)


models = {'LogisticRegression':lr,
		  'DecistionTree':dt,
		  'RandomForest':rf}

modelPerf = {k:testModel(v) for k,v in models.iteritems()}
