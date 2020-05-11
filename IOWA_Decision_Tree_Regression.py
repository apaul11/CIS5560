# Databricks notebook source
# MAGIC %md ## CIS5560: IOWA - Decision Tree Regression
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
from pyspark.ml.classification import LogisticRegression


from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

import sys


# COMMAND ----------

# MAGIC %md ### TODO 0: Run the code in PySpark CLI
# MAGIC 1. Set the following to True:
# MAGIC ```
# MAGIC PYSPARK_CLI = True
# MAGIC ```
# MAGIC 1. We need to generate py (Python) file: File > Export > Source File
# MAGIC 1. Run it at our Hadoop/Spark cluster:
# MAGIC ```
# MAGIC $ spark-submit IOWA_Decision_Tree_Regression.py
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

# In decision tree regression
dtr_train = splits[0]
dtr_test = splits[1].withColumnRenamed("label", "trueLabel")

print ("DTR Training Rows:", dtr_train.count(), "DTR Testing Rows:", dtr_test.count())

dtr_train.show(20)

# COMMAND ----------

# MAGIC %md ### Define the Pipeline and Train the Model
# MAGIC Now define a pipeline that creates a feature vector and trains a regression model

# COMMAND ----------

# Define the pipeline
#Use DecisionTreeRegressor in Decision Tree Regression

assembler = VectorAssembler(inputCols = ["Pack", "BottleVolumeInMl","StateBottleCost", "StateBottleRetail", "BottlesSold", "VolumeSoldInLitres"], outputCol="features")

dtr = DecisionTreeRegressor(featuresCol='features', labelCol='label', maxBins=77582)

dtr_pipeline = Pipeline(stages=[assembler, dtr])

# COMMAND ----------

# MAGIC %md ##Define the ParameterGrid and Tune the Parameters
# MAGIC In ParameterGrid, we define the parameters maxDepth, minInfoGain, stepSize. Then we use TrainValidationSplit to evaluate each combination of the parameters defined in the ParameterGrid.

# COMMAND ----------

#paramGridDtr = ParamGridBuilder().build()

paramGridDtr = ParamGridBuilder().build()

#cvDtr = CrossValidator(estimator=gbt_pipeline, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGridDtr, numFolds=5)

cvDtr = CrossValidator(estimator=dtr_pipeline, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGridDtr, numFolds=5)

# COMMAND ----------

# MAGIC %md ##Train the model

# COMMAND ----------

dtr_model = cvDtr.fit(dtr_train)

# COMMAND ----------

# MAGIC %md ### Test the Model
# MAGIC Now you're ready to apply the model to the test data.

# COMMAND ----------

dtr_prediction = dtr_model.transform(dtr_test)
dtr_predicted = dtr_prediction.select("features", "prediction", "trueLabel")
dtr_predicted.show(20)

# COMMAND ----------

# MAGIC %md ### Examine the Predicted and Actual Values

# COMMAND ----------

dtr_predicted.createOrReplaceTempView("regressionPredictions")

# COMMAND ----------

# MAGIC %md ### data visualization using SQL in Databricks.

# COMMAND ----------

# MAGIC %md ### TODO 1: Visualize the following sql as scatter plot. 
# MAGIC 1. Then, select the icon graph "Show in Dashboard Menu" in the right top of the cell to create a Dashboard
# MAGIC 1. Select "+Add to New Dashboard" and will move to new web page with the scatter plot chart
# MAGIC 1. Name the dashboard to __Decision Tree Regression__
# MAGIC 
# MAGIC __NOTE__: _%sql_ does not work at PySpark CLI but only at Databricks notebook.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT trueLabel, prediction FROM regressionPredictions

# COMMAND ----------

# MAGIC %md ### Retrieve the Root Mean Square Error (RMSE)
# MAGIC There are a number of metrics used to measure the variance between predicted and actual values. Of these, the root mean square error (RMSE) is a commonly used value that is measured in the same units as the predicted and actual values - so in this case, the RMSE indicates the average amounts between predicted and actual sale values. We can use the **RegressionEvaluator** class to retrieve the RMSE.

# COMMAND ----------

dtr_evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
dtr_rmse = dtr_evaluator.evaluate(dtr_prediction)
print ("Root Mean Square Error (RMSE_DTR):", dtr_rmse)
