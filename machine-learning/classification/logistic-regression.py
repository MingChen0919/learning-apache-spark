from __future__ import print_function
from pyspark import SparkConf, SparkContext

import os
import pandas as pd
from ggplot import *

from pyspark.sql import SQLContext
from pyspark.sql.types import *


os.chdir('/Users/mingchen/GoogleDrive/R-projects/learning-apache-spark/machine-learning/classification/')
os.getcwd()

## set up spark context
conf = SparkConf().setAppName("logistic-regression")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)



'''
horseshoeCrab = sc.textFile('horseshoe_crab.csv')

## delete the first row (table header)
header = horseshoeCrab.first()
horseshoeCrab = horseshoeCrab.filter(lambda x: x != header)
## convert rows to lists
horseshoeCrab = horseshoeCrab.map(lambda x: x.strip().split(','))
'''




horseshoeCrab = sqlContext.read.load('horseshoe_crab.csv',
                      format='com.databricks.spark.csv', 
                      header='true', 
                      inferSchema='true')

horseshoeCrab.select('W').take(10)

dir(horseshoeCrab)