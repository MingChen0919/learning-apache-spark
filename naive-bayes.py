## connect to spark
from pyspark import SparkConf, SparkContext
## set up spark context
conf = SparkConf().setAppName("myApp")
sc = SparkContext(conf=conf)

# create sparksession object
from pyspark.sql import SparkSession
sparksession = SparkSession(sc)




## Import Data

wine_data = sparksession.read.csv("./data/WineData.csv", inferSchema=True, header=True)
wine_data.printSchema()
wine_data.show(5)





## # process categorical columns
## from pyspark.ml.feature import StringIndexer
## indexer = StringIndexer(inputCol="quality", outputCol="indexed_label")
## model = indexer.fit(wine_data)
## wine_data_label_indexed = model.transform(wine_data)
## wine_data_label_indexed.show(5)
## #wine_data = wine_data.withColumn('string_quality', wine_data.quality.cast('string'))
## 
## 
## from pyspark.ml.feature import StringIndexer
## indexer = StringIndexer(inputCol="label", outputCol="indexed_label")
## model = indexer.fit(ml_wine_data)
## wine_data_label_indexed = model.transform(ml_wine_data)
## wine_data_label_indexed.show(5)


# convert data into featuresCol and labelCol structre
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
ml_wine_data = wine_data.rdd.map(lambda r: [Vectors.dense(r[:-1]), r[-1]]).toDF(['featuresCol', 'label'])
ml_wine_data.show(5)



## from pyspark.ml.feature import VectorIndexer
## indexer = VectorIndexer(maxCategories=4, inputCol='featuresCol', outputCol='indexed_features')
## model = indexer.fit(wine_data_label_indexed)
## wine_data_feature_indexed  = model.transform(wine_data_label_indexed)

## splitting data into training and test sets
training, test = ml_wine_data.randomSplit(weights=[0.7, 0.3], seed=123)
training.show(5)


## naive bayes classifier
## logistic regression classifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier

nb = NaiveBayes(featuresCol="featuresCol", labelCol="label")
mlor = LogisticRegression(featuresCol='indexed_features', labelCol='label')
dt = DecisionTreeClassifier(featuresCol="indexed_features", labelCol="label")
rf = RandomForestClassifier(featuresCol="indexed_features", labelCol="label")
gbt = GBTClassifier(featuresCol="indexed_features", labelCol="label")
mpnn = MultilayerPerceptronClassifier(featuresCol="indexed_features", labelCol="label")


nb.fit(training).transform(training).select(['prediction']).distinct().show()
dt.fit(training).transform(training).select(['prediction']).distinct().show()
evaluator.evaluate(nb.fit(training).transform(training))


# build parameter grid
from pyspark.ml.tuning import ParamGridBuilder
# param grid for naive bayes
nb_param_grid = ParamGridBuilder().\
    addGrid(nb.smoothing, [0, 0.5, 1, 2, 5, 10]).\
    build()
# param grid for logistic regression
mlor_param_grid = ParamGridBuilder().\
    addGrid(mlor.regParam, [0, 0.5, 1, 2]).\
    addGrid(mlor.elasticNetParam, [0, 0.5, 1]).\
    build()
# param grid for decision tree
dt_param_grid = ParamGridBuilder().\
    addGrid(dt.minInfoGain, [0,1,2,3]).\
    build()
# param_grid for random forest
rf_param_grid = ParamGridBuilder().\
    addGrid(rf.minInfoGain, [0,1,2,3]).\
    build()
# param_grid for Gradient-Boosted Trees
gbt_param_grid = ParamGridBuilder().\
    addGrid(gbt.minInfoGain, [0,1,2,3]).\
    build()
# param_grid for Gradient-Boosted Trees


# input_neurons is number of features
input_neurons = len(training.select(['indexed_features']).take(1)[0]['indexed_features'])
input_neurons
output_neuros = len(training.select(['indexed_label']).distinct().collect())
output_neuros
mpnn_param_grid = ParamGridBuilder().\
    addGrid(mpnn.layers, [(input_neurons, output_neuros), (input_neurons, 10, output_neuros), (input_neurons, 20, output_neuros)]).\
    build()

training.show(5)

# classifier evaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator()

# run k (k=4) folds cross validation
from pyspark.ml.tuning import CrossValidator
nb_cv = CrossValidator(estimator=nb, estimatorParamMaps=nb_param_grid, evaluator=evaluator, numFolds=4)
mlor_cv = CrossValidator(estimator=mlor, estimatorParamMaps=mlor_param_grid, evaluator=evaluator, numFolds=4)
dt_cv = CrossValidator(estimator=dt, estimatorParamMaps=dt_param_grid, evaluator=evaluator, numFolds=4)
rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_param_grid, evaluator=evaluator, numFolds=4)
gbt_cv = CrossValidator(estimator=gbt, estimatorParamMaps=gbt_param_grid, evaluator=evaluator, numFolds=4)
mpnn_cv = CrossValidator(estimator=mpnn, estimatorParamMaps=mpnn_param_grid, evaluator=evaluator, numFolds=4)


new_training = training.drop('label')
new_training = new_training.withColumnRenamed('indexed_label', 'label')
new_training = new_training.withColumn('indexed_label', new_training.label)
new_training.show(5)

# fit model
nb_cv_model = nb_cv.fit(training)
mlor_cv_model = mlor_cv.fit(training)
dt_cv_model = dt_cv.fit(training)
rf_cv_model = rf_cv.fit(training)
#gbt_cv_model = gbt_cv.fit(training)
mpnn_cv_model = mpnn_cv.fit(training)


evaluator.evaluate(nb_cv_model.transform(training))
evaluator.evaluate(nb_cv_model.transform(training))
evaluator.evaluate(mlor_cv_model.transform(training))
evaluator.evaluate(dt_cv_model.transform(training))
evaluator.evaluate(rf_cv_model.transform(training))
evaluator.evaluate(mpnn_cv_model.transform(training))


evaluator.evaluate(nb_cv_model.transform(new_test))
evaluator.evaluate(mlor_cv_model.transform(new_test))
evaluator.evaluate(dt_cv_model.transform(new_test))
evaluator.evaluate(rf_cv_model.transform(new_test))
evaluator.evaluate(mpnn_cv_model.transform(new_test))

# classifier evaluation (training)
nb_tr = 1 - evaluator.evaluate(nb_cv_model.transform(new_training))
mlor_tr = 1- evaluator.evaluate(mlor_cv_model.transform(new_training))
dt_tr = 1 - evaluator.evaluate(dt_cv_model.transform(new_training))
rf_tr = 1 - evaluator.evaluate(rf_cv_model.transform(new_training))
mpnn_tr = 1- evaluator.evaluate(mpnn_cv_model.transform(new_training))

new_test = test.drop('label')
new_test = new_test.withColumn('label', new_test.indexed_label)
new_test.show(5)
# classifier evaluation (test)
nb_testerr = 1 - evaluator.evaluate(nb_cv_model.transform(new_test))
mlor_testerr = 1 - evaluator.evaluate(mlor_cv_model.transform(new_test))
dt_testerr = 1 - evaluator.evaluate(dt_cv_model.transform(new_test))
rf_testerr = 1 - evaluator.evaluate(rf_cv_model.transform(new_test))
mpnn_testerr = 1 - evaluator.evaluate(mpnn_cv_model.transform(new_test))



print("naive bayes training error %g \n" % nb_tr
      "naive bayes test error %g \n" % nb_testerr)
print("naive bayes trainig and test error: %g", % (nb_tr, nb_testerr))

import pandas as pd
pd.DataFrame(
    {
        "training error": [nb_tr, mlor_tr, dt_tr, rf_tr, mpnn_tr],
        "test error": [nb_testerr, mlor_testerr, dt_testerr, rf_testerr, mpnn_testerr]
    },
    index = ["naive bayes", "logistic regression", "decision tree", "random forest", "neural network"]
    )



