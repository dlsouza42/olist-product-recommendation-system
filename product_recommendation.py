# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator


# Creating Spark Session


sc = pyspark.SparkContext(appName="customer")

spark = SparkSession.builder.getOrCreate()


#conf = pyspark.SparkConf().setAll([('spark.executor.memory', '8g'), ('spark.executor.cores', '4'), ('spark.cores.max', '4'), ('spark.driver.memory','8g')])

#sc = pyspark.SparkContext()
#spark = SparkSession.builder.getOrCreate()

# Importing data

items = spark.read.csv("OneDrive\Área de Trabalho\Olist Data\olist_order_items_dataset.csv", header = True )


review = pd.read_csv("OneDrive\Área de Trabalho\Olist Data\olist_order_reviews_dataset.csv", index_col=None)
review = review.iloc[:, 0:3]
review = spark.createDataFrame(review)

orders = spark.read.csv("OneDrive\Área de Trabalho\Olist Data\olist_orders_dataset.csv", header = True )
products = spark.read.csv("OneDrive\Área de Trabalho\Olist Data\olist_products_dataset.csv", header = True )

# Data Manipulation ---------

# Select Columns From items

items = items.select('order_id', 'product_id').distinct()


# Select Columns From Order

orders = orders.select('order_id', 'customer_id').distinct()

# Joining with Review

data = review.join(items, on = 'order_id', how='left').join(orders, on = 'order_id', how='left')

# Dropping other columns

data = data.drop(*['review_id', 'order_id'])

# See data schema

data.printSchema()

# Creating a new customer id

consumers = data.select('customer_id')

consumers = consumers.coalesce(1)

consumers = consumers.withColumn("userIntId", monotonically_increasing_id()).persist()


# Creating a new product id

product = data.select('product_id')

product = product.coalesce(1)

product = product.withColumn('productIntId', monotonically_increasing_id()).persist()

# Integrating new columns with 

data = data.join(consumers, on = 'customer_id', how='left').join(product, on = 'product_id', how='left')

# Creating final data

review_data = data.select('userIntId', 'productIntId', 'review_score').\
    withColumnRenamed('userIntId', 'customer_id').withColumnRenamed('productIntId', 'product_id')
    
# Calculating sparsity

# - Number of products

p = review_data.select('product_id').distinct().count()
c = review_data.select('customer_id').distinct().count()

sparsity = review_data.count()/(p*c)
sparsity = 1-sparsity

print ("Sparsity: ", sparsity)

del p, c, sparsity, items, orders, review, data

# Model Creation -----------------------------------------------------------

review_data = review_data.na.drop()



# Create training and test set (80/20 split)


(training, test) = review_data.randomSplit([0.8, 0.2])

# Build generic ALS model without hyperparameters

als = ALS(userCol="customer_id", itemCol="product_id", 
          ratingCol="review_score",coldStartStrategy="drop",
          nonnegative = True, implicitPrefs = False)

# Build Parameter Grid

param_grid = ParamGridBuilder().addGrid(als.rank, [25, 40]).\
    addGrid(als.maxIter, [250]).\
        addGrid(als.regParam, [.05, .1, .5]).build()  
        
# Build Evaluator

evaluator = RegressionEvaluator(metricName="rmse", labelCol="review_score", predictionCol="prediction")

# Build Cross Validator

cv = CrossValidator(estimator = als,
                    estimatorParamMaps = param_grid,
                    evaluator = evaluator,    
                    numFolds = 5)    

# Run the cv on the training data               
   
model = cv.fit(training)    

training.show()
