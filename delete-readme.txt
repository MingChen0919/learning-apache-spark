# Learning apache spark

**[Ming Chen](https://github.com/MingChen0919) & [Wenqiang Feng](http://web.utk.edu/~wfeng1/)**

## Introduction

This repository contains mainly notes from learning Apache Spark by [Ming Chen](https://github.com/MingChen0919) & [Wenqiang Feng](http://web.utk.edu/~wfeng1/). We try to use the detailed demo code and examples to show how to use pyspark for big data mining. **If you find your work wasn't cited in this note, please feel free to let us know.**

## Content

* ***Cheat Sheets***
    + [Python Basics](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PythonForDataScience.pdf)
    + [Pandas Basics](http://datacamp-community.s3.amazonaws.com/3857975e-e12f-406a-b3e8-7d627217e952)
    + [Numpy Basics](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
    + [Scikit-Learn](http://datacamp-community.s3.amazonaws.com/5433fa18-9f43-44cc-b228-74672efcd116)
    + [RDD Basics](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_Cheat_Sheet_Python.pdf)
    + [DataFrame Basics](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_SQL_Cheat_Sheet_Python.pdf)
    + [Apache Spark Cheat Sheet](https://hackr.io/tutorials/learn-apache-spark)
    
* ***Data Manipulation***
    + [Entry Points to Spark](entry-points-to-spark.ipynb)
    + [RDD Object](rdd-object.ipynb)
    + [DataFrame Object](dataframe-object.ipynb)
    + [RDD and DataFrame conversion](conversion-between-rdd-and-dataframe.ipynb)
    + [Categorical Data, `StringIndexer` and `OneHotEncoder`](stringindexer-and-onehotencoder.ipynb)
    + [Continuous variables to categorical variables](Continuous-variable-to-categorical-variable.ipynb)
    + [Import and export data](import-and-export-data.ipynb)   
    + [Subset data](subset-data.ipynb):
        * select rows by index
        * select rows by logical criteria
        * select columns by index
        * select columns by names
        * select columns by regex pattern
    + [`udf()` function and SQL data types](udf-and-sql-types.ipynb):
        * use `udf()` function
        * difference between `ArrayType` and `StructType`
    + [Pipeline](pipeline.ipynb)
    + [Dense and sparse vectors](dense-vs-sparse-vectors.ipynb)
    + [Assemble feature columns into a `featuresCol` column with `VectorAssembler`](vector-assembler.ipynb)
    + [TF-IDF, HashingTF and CountVectorizer](TF-IDF.ipynb)
    + Feature processing:
    	- [First data check](first-data-check.ipynb)
    + [SQL functions](sql-functions.ipynb)
    + [Add py Files to cluster](add-py-files-to-spark-cluster.ipynb)

* ***Machine Learning***
    + [Machine Learning Framework](machine-learning-framework.Rmd)
    + **Regression**

        - [Linear regression](linear-regression.ipynb)
        - [Logistic regression](logistic-regression.ipynb)
    
    + **Classification**

		- [Naive bayes classification](naive-bayes-classification.ipynb)
		- [Decision tree](decision-tree-classification.ipynb)
		- [Random forest classification](random-forest-classification.ipynb)
		- [Gradient boost tree classification](gradient-boost-tree-classification.ipynb)
    
* **Model Tuning**
    + [Regularization](regularization.ipynb)
    + [Cross-validation](cross-validation.ipynb)

* **Nutural Language Processing**
    + [NLP and NLTK Basics](nlp-and-nltk-basics.ipynb)
    + [NLP Information Extraction](nlp-information-extraction.ipynb)
    
### Acknowledgement

At here, we would like to thank Jian Sun and Zhongbo Li at the University of Tennessee at Knoxville for the valuable disscussion and thank the generous anonymous authors for providing the detailed solutions and source code on the internet. Without those help, this repository would not have been possible to be made. Wenqiang also would like to thank the Institute for Mathematics and Its Applications (IMA) at University of Minnesota, Twin Cities for support during his IMA Data Scientist Fellow visit. 

### Feedback and suggestions

Your comments and suggestions are highly appreciated. We are more than happy to receive corrections, suggestions or feedbacks through email (Ming Chen: mchen33@utk.edu, Wenqiang Feng: wfeng1@utk.edu) for improvements.
