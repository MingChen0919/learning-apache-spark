from __future__ import print_function
from pyspark import SparkConf, SparkContext
from collections import OrderedDict

from pyspark.mllib.clustering import KMeans, KMeansModel
#from pyspark.mllib.linalg import DenseVector
#from pyspark.mllib.linalg import SparseVector

import os
from ggplot import *



os.chdir('/Users/mingchen/GoogleDrive/R-projects/learning-apache-spark')
os.getcwd()



## set up spark context
conf = SparkConf().setAppName("myApp")
sc = SparkContext(conf=conf)


myData = sc.parallelize([(1,2), (3,4), (5,6), (7,8), (9,10)])
myData.collect()