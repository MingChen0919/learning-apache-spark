# logistic regression

# connecting to spark
from pyspark import SparkConf, SparkContext
## set up spark context
conf = SparkConf().setAppName("myApp")
sc = SparkContext(conf=conf)

# create sparksession object
from pyspark.sql import SparkSession
sparksession = SparkSession(sc)

# import data
# read data from csv
# iris = sparksession.read.csv("data/iris.csv", inferSchema=True, header=True)
mtcars = sparksession.read.csv("data/mtcars.csv", inferSchema=True, header=True)
mtcars.show()

## select columns by column names
mtcars.select(['mpg', 'cyl']).show()
## select columns by indices
new_mtcars = mtcars.rdd.map(lambda x: [x[i] for i in range(1,6)]).toDF()

## reset column names
new_column_names = [mtcars.columns[i] for i in range(1, 6)]
ori_column_names = new_mtcars.columns
column_names = zip(ori_column_names, new_column_names)
for i in range(0, len(new_mtcars.columns)):
    new_mtcars = new_mtcars.withColumnRenamed(column_names[i][0], column_names[i][1])

## comparison
new_mtcars.show(n=5)
mtcars.show(n=5)


new_column_names
ori_column_names


mtcars.rdd.map(lambda x: [x[i] for i in (1,3,6,7)]).toDF().show()


