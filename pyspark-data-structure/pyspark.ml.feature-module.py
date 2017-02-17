# pyspark.ml.feature module

#
from pyspark.ml.feature import Binarizer
df = sparksession.createDataFrame([(0.5,)], ["values"])
df.collect()
binarizer = Binarizer(threshold=1.0, inputCol="values", outputCol="features")
df2 = binarizer.transform(df)
df2.dtypes
df.collect()
df2.collect()
binarizer.getOutputCol()

rawData.take(1)
binarizer2 = Binarizer(threshold=0.5, inputCol="srv_diff_host_rate", outputCol="features")
binarizer2.transform(rawData)

binarizer.explainParam('inputCol')
binarizer.inputCol
binarizer.params

rawData.select(['count']).show()


rawData.dtypes
from pyspark.ml.feature import StringIndexer
stringIndexer = StringIndexer(inputCol="y_label", outputCol='indexed_y_label')
model = stringIndexer.fit(rawData)
td = model.transform(rawData)
td.dtypes

td.select([td.y_label, td.indexed_y_label]).show()



df = sparksession.createDataFrame([(Vectors.dense([0.0]),), (Vectors.dense([2.0]),)], ["a"])
