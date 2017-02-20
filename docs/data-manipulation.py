# data manipulation

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
mtcars.show(5)

## select columns by column names
mtcars.select(['mpg', 'cyl']).show(n=5)
## select columns by indices
new_mtcars = mtcars.rdd.map(lambda x: [x[i] for i in range(1,6)]).toDF()
new_mtcars.show(5)

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


mtcars.rdd.map(lambda x: [x[i] for i in range(1,6)]).toDF().show(n=5)
mtcars.rdd.map(lambda x: [x[i] for i in (1,3,6,7)]).toDF().show(n=5)

mtcars
mtcars.withColumnRenamed('mpg', 'new_mpg').show(n=5)
mtcars.show(n=5)
.withCoumnRenamed('mpg', 'new_mpg').show(n=5)


## get column number
len(mtcars.columns)
mtcars.count()

mtcars.select(['carb']).distinct().show()

training, test = mtcars.randomSplit(weights=[0.75, 0.25], seed=24)
training.show()
test.show()

df1, df2, df3, df4 = mtcars.randomSplit(weights=[0.2, 0.2, 0.15, 0.25], seed=123)
df1.count()
df2.count()
df3.count()
df4.count()


mtcars.show()

iris.show(n=5)

from pyspark.ml.feature import VectorIndexer
iris.select(['species']).distinct().count()
indexer = VectorIndexer(maxCategories=3, inputCol='species', outputCol='indexed_species')
model = indexer.fit(iris)


from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol='species', outputCol='indexed_species')
model = indexer.fit(iris)
model.transform(iris).show(n=5)
model.transform(iris).select(['species', 'indexed_species']).distinct().show()


species = iris.select(['species'])

mtcars.show(n=5)

mtcars.select(['carb']).distinct().count()

indexer = VectorIndexer(maxCategories=6, inputCol='carb', outputCol='indexed_carb')
model = indexer.fit(mtcars)

mtcars.dtypes
mtcars.schema

from pyspark.ml.linalg import Vectors
df = sparksession.createDataFrame([(Vectors.dense([-1.0, 0.0]),), (Vectors.dense([0.0, 1.0]),), (Vectors.dense([0.0, 2.0]),)], ["a"])
df.show()
indexer = VectorIndexer(maxCategories=2, inputCol="a", outputCol="indexed")
model = indexer.fit(df)
model.transform(df).show()



advertising = sparksession.read.csv("data/Advertising.csv", inferSchema=True, header=True)

advertising.show(n=5)

from pyspark.ml.linalg import Vectors
from pyspark.sql import Row

transformed_data = advertising.rdd.map(lambda x: Row(featurs=Vectors.dense[x[0:3]], label=x[3])).toDF()


# convert the data to dense vector
def transData(row):
    return Row(label=row["Sales"],
               features=Vectors.dense([
                                       row["TV"],
                                       row["Radio"],
                                       row["Newspaper"]]))
                                       
transformed = advertising.rdd.map(transData).toDF() 

]
from pyspark.ml.regression import LinearRegression
lr = LinearRegression()
model = lr.fit(transformed)
model.transform(transformed).show(n=5)


hsb2 = sparksession.read.csv("data/hsb2_modified.csv", inferSchema=True, header=True)
hsb2.show(n=5)

    
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol='gender', outputCol='indexed_gender')
model = indexer.fit(hsb2)
hsb2 = model.transform(hsb2)
hsb2.show(n=5)


new_hsb2 = hsb2.rdd.map(lambda row: Row(label=row['socst'],
                                        features=Vectors.dense([
                                            row['indexed_gender'],
                                            row['race'],
                                            row['ses'],
                                            row['schtyp'],
                                            row['prog'],
                                            row['read'],
                                            row['write'],
                                            row['math'],
                                            row['science']])
                                        )).toDF()
new_hsb2.show(n=5)

from pyspark.ml.feature import VectorIndexer
indexer = VectorIndexer(maxCategories=4, inputCol='features', outputCol='indexed_features')
model = indexer.fit(new_hsb2)
indexed_new_hsb2 = model.transform(new_hsb2)
indexed_new_hsb2.show(n=5)


from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol='indexed_features', labelCol='label')
model = lr.fit(indexed_new_hsb2)
model.transform(indexed_new_hsb2).show()




horseshoe_crab = sparksession.read.csv("data/horseshoe_crab.csv", inferSchema=True, header=True)
horseshoe_crab.show(n=5)

def binary_converter(x):
    if x > 0:
        x = 1
    else:
        x = 0
    return x
    
horseshoe_crab.withColumn('new_Sa', bool(horseshoe_crab.Sa)).show(n=5)

dir(horseshoe_crab.Sa)