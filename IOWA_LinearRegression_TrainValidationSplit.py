# Databricks notebook source
# MAGIC %md ## CIS5560: IOWA - Linear Regression + Cross Validation + Parameter Tuning using TrainValidationSplit
# MAGIC 
# MAGIC ### Project 5560

# COMMAND ----------

# MAGIC %md ##Import the Libraries
# MAGIC 
# MAGIC First, import the libraries we will need to create the dataframe and make a sample out of it.

# COMMAND ----------

# Import Spark SQL and Spark ML libraries

from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.regression import LinearRegression


from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

import sys


# COMMAND ----------

# MAGIC %md ### TODO 0: Run the code in PySpark CLI
# MAGIC 1. Set the following to True:
# MAGIC ```
# MAGIC PYSPARK_CLI = True
# MAGIC ```
# MAGIC 1. You need to generate py (Python) file: File > Export > Source File
# MAGIC 1. Run it at your Hadoop/Spark cluster:
# MAGIC ```
# MAGIC $ spark-submit IOWA_LinearRegression+CrossValidation+TrainValidationSplit.py
# MAGIC ```

# COMMAND ----------

PYSPARK_CLI = True
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# DataFrame Schema
liquorsalesSchema = StructType([
  StructField("Invoice/Item Number", StringType(), False),
  StructField("Date", StringType(), False),
  StructField("StoreNumber", IntegerType(), False),
  StructField("StoreName", StringType(), False),
  StructField("Address", StringType(), False),
  StructField("City", StringType(), False),
  StructField("ZipCode", IntegerType(), False),
  StructField("StoreLocation", StringType(), False),
  StructField("CountyNumber", IntegerType(), False),
  StructField("County", StringType(), False),
  StructField("Category", IntegerType(), False),
  StructField("CategoryName", StringType(), False),
  StructField("VendorNumber", IntegerType(), False),
  StructField("VendorName", StringType(), False),
  StructField("ItemNumber", IntegerType(), False),
  StructField("ItemDescription", StringType(), False),
  StructField("Pack", IntegerType(), False),
  StructField("BottleVolumeInMl)", IntegerType(), False),
  StructField("StateBottleCost", DoubleType(), False),
  StructField("StateBottleRetail", DoubleType(), False),
  StructField("BottlesSold", IntegerType(), False),
  StructField("SaleInDollars", DoubleType(), False),
  StructField("VolumeSoldInLitres",DoubleType(), False),
  StructField("VolumeSoldInGallons", DoubleType(), False),
])

# COMMAND ----------

# MAGIC %md ### Load the Data to the table

# COMMAND ----------

if PYSPARK_CLI:
    csv = spark.read.csv('Iowa_Liquor_Sales_Cleaned_sample.csv', inferSchema=True, header=True)
else:
   csv = spark.sql("SELECT * FROM iowaliquorsalessample_csv")

# Load the source data
# csv = spark.read.csv('wasb:///data/iowa_Liquor_Sales.csv', inferSchema=True, header=True)

csv.show(truncate = False)

# COMMAND ----------

# MAGIC %md ##Select features and label
# MAGIC 
# MAGIC ####Select the relevant columns in a new dataframe and define the label.

# COMMAND ----------

# Select relevant columns.
csv1 = csv.select("Pack", "BottleVolumeInMl", "StateBottleCost", "StateBottleRetail", "BottlesSold", "SaleInDollars", "VolumeSoldInLitres")

df1 = csv1.filter(csv1.StateBottleCost.isNotNull())
df2 = df1.filter(df1.StateBottleRetail.isNotNull())
df3 = df2.filter(df2.BottleVolumeInMl.isNotNull())
df4 = df3.filter(df3.Pack.isNotNull())
df5 = df4.filter(df4.BottlesSold.isNotNull())
df6 = df5.filter(df5.SaleInDollars.isNotNull())
df7 = df6.filter(df6.VolumeSoldInLitres.isNotNull())

# Select features and label
data = df7.select(col("Pack").cast(DoubleType()), col("BottleVolumeInMl").cast(DoubleType()), "StateBottleCost", "StateBottleRetail", col("BottlesSold").cast(DoubleType()), "VolumeSoldInLitres", col("SaleInDollars").alias("label"))

data.show(5)

# COMMAND ----------

# MAGIC %md ##Split the data
# MAGIC ####Split the data in 70-30 train-test ratio.

# COMMAND ----------

# Split the data
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")

# COMMAND ----------

# MAGIC %md ### Define the Pipeline and Train the Model
# MAGIC Now define a pipeline that creates a feature vector and trains a regression model

# COMMAND ----------

# Define the pipeline
assembler = VectorAssembler(inputCols = ["Pack", "BottleVolumeInMl","StateBottleCost", "StateBottleRetail", "BottlesSold", "VolumeSoldInLitres"], outputCol="features")

lr = LinearRegression(labelCol="label",featuresCol="features", maxIter=10, regParam=0.3)

pipeline = Pipeline(stages=[assembler, lr])


# COMMAND ----------

# MAGIC %md ### Tune Parameters
# MAGIC We can tune parameters to find the best model for our data. A simple way to do this is to use  **TrainValidationSplit** to evaluate each combination of parameters defined in a **ParameterGrid** against a subset of the training data in order to find the best performing parameters. We are also using CrossValidator class to evaluate each combination of parameters defined in a **ParameterGrid** against multiple *folds* of the data split into training and validation datasets, in order to find the best performing parameters. Note that this can take a long time to run because every parameter combination is tried multiple times.
# MAGIC 
# MAGIC #### Regularization 
# MAGIC is a way of avoiding Imbalances in the way that the data is trained against the training data so that the model ends up being over fit to the training data. In other words It works really well with the training data but it doesn't generalize well with other data.
# MAGIC That we can use a **regularization parameter** to vary the way that the model balances that way.
# MAGIC 
# MAGIC #### Training ratio of 0.8
# MAGIC it's going to use 80% of the the data that it's got in its training set to train the model and then the remaining 20% is going to use to validate the trained model. 
# MAGIC 
# MAGIC In **ParamGridBuilder**, all possible combinations are generated from regParam, maxIter. So it is going to try each combination of the parameters with 80% of the the data to train the model and 20% to to validate it.



# COMMAND ----------

# MAGIC %md ##TrainValidationSplit

# COMMAND ----------

paramGrid_tvs = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.1, 0.01]).addGrid(lr.maxIter, [10, 5]).build()

tvs = TrainValidationSplit(estimator=pipeline, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid_tvs, trainRatio=0.8)

piplineModel_tvs = tvs.fit(train)

# COMMAND ----------

# MAGIC %md ### Test the Model with TrainValidationSplit
# MAGIC Now we are ready to apply the model to the test data.

# COMMAND ----------

prediction = piplineModel_tvs.transform(test)
predicted_tvs = prediction.select("features", "prediction", "trueLabel")
predicted_tvs.show(10)

# COMMAND ----------

# MAGIC %md ### Examine the Predicted and Actual Values

# COMMAND ----------

predicted_tvs.createOrReplaceTempView("regressionPredictionsTVS")

# COMMAND ----------

# MAGIC %md ### data visualization using SQL in Databricks.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT trueLabel, prediction FROM regressionPredictionsTVS

# COMMAND ----------

# MAGIC %md ### Retrieve the Root Mean Square Error (RMSE)
# MAGIC There are a number of metrics used to measure the variance between predicted and actual values. Of these, the root mean square error (RMSE) is a commonly used value that is measured in the same units as the predicted and actual values - so in this case, the RMSE indicates the average amounts between predicted and actual sale values. We can use the **RegressionEvaluator** class to retrieve the RMSE.

# COMMAND ----------


evaluator_tvs = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")

rmse = evaluator_tvs.evaluate(prediction)

print ("Root Mean Square Error (RMSE_tvs):", rmse)
