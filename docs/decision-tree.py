## connect to spark
from pyspark import SparkConf, SparkContext
## set up spark context
conf = SparkConf().setAppName("myApp")
sc = SparkContext(conf=conf)

# create SparkSession object
from pyspark.sql import SparkSession
sparksession = SparkSession(sc)


# import data
rawData = sparksession.read.csv("./data/WineData.csv", inferSchema=True, header=True)
rawData.show(n=5)

# convert data to ML structure
from pyspark.ml.linalg import Vectors
df = rawData.rdd.map(lambda r: [Vectors.dense(r[:-1]), r[-1]]).toDF(['features', 'label'])

# index labels
from pyspark.ml.feature import IndexToString,StringIndexer
labelIndexer = StringIndexer(inputCol='label',
                             outputCol='indexedLabel').fit(df)

# index features
from pyspark.ml.feature import VectorIndexer
featureIndexer = VectorIndexer(inputCol='features',
                               outputCol='indexedFeatures').fit(df)

# build decision tree model
from pyspark.ml.classification import DecisionTreeClassifier
dTree = DecisionTreeClassifier(featuresCol='indexedFeatures', labelCol='indexedLabel')


# convert predicted indices back to labels
labelConverter = IndexToString(inputCol='prediction',
                               outputCol='predictedLabel',
                               labels=labelIndexer.labels)
                        
# chain indexers and model in a pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dTree, labelConverter])

# split the data into training and test sets
(training, test) = df.randomSplit(weights=[0.8, 0.2])

# train model
model = pipeline.fit(training)

# prediction with training data
prediction_training = model.transform(training)
# prediction with test data
prediction_test = model.transform(test)

## cross validation
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
paramGrid = ParamGridBuilder().\
    addGrid(dTree.minInfoGain, [0,1,2]).\
    addGrid(dTree.maxDepth, [2,5,10]).\
    build()

# evaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol='indexedLabel',
                                              predictionCol='prediction')

# 3-fold cross validation
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)

# train model through cross validation
dTree_cv_model = cv.fit(training)

# test model
dTree_prediction = dTree_cv_model.transform(training)
dTree_prediction_test = dTree_cv_model.transform(test)
dTree_prediction.show()

# evaluate model
evaluator.evaluate(dTree_prediction)
evaluator.evaluate(dTree_prediction_test)

       
                               
