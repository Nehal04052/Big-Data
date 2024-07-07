# Databricks notebook source
# MAGIC %md
# MAGIC # Ice-cream Revenue Prediction
# MAGIC - Independant variable X: Outside Air Temperature
# MAGIC - Dependant variable Y: Overall daily revenue generated in dollars

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/tables/")

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, abs
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

spark = SparkSession.builder.appName('Ice-cream Revenue Prediction').getOrCreate() 

# COMMAND ----------

spark

# COMMAND ----------

df_pyspark = spark.read.csv('dbfs:/FileStore/tables/Ice_Cream.csv',header=True,inferSchema=True)

# COMMAND ----------

df_pyspark.printSchema()

# COMMAND ----------

df_pyspark

# COMMAND ----------

df_pyspark.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Clean the DataFrame

# COMMAND ----------

# Handle missing values if necessary
df_pyspark = df_pyspark.na.drop()

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare the DataFrame

# COMMAND ----------

# Define the feature columns
feature_columns = ["Temperature"]

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_pyspark = assembler.transform(df_pyspark)

# Select only the features and target column
df_pyspark = df_pyspark.select("features", "Revenue")

# COMMAND ----------

df_pyspark.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Split the DataFrame

# COMMAND ----------

# Split the data into training and testing sets
train_data, test_data = df_pyspark.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Train the Model

# COMMAND ----------

# Initialize the linear regression model
lr = LinearRegression(featuresCol="features", labelCol="Revenue")

# Fit the model on the training data
lr_model = lr.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Evaluate the Model

# COMMAND ----------

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Evaluate the model using RMSE
rmse_evaluator = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="rmse")
rmse = rmse_evaluator.evaluate(predictions)

# Evaluate the model using MAE
mae_evaluator = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="mae")
mae = mae_evaluator.evaluate(predictions)

# Evaluate the model using MSE
mse_evaluator = RegressionEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="mse")
mse = mse_evaluator.evaluate(predictions)

# Calculate MAPE (Mean Absolute Percentage Error)
predictions = predictions.withColumn("absolute_error", abs(col("prediction") - col("Revenue")))
predictions = predictions.withColumn("percentage_error", col("absolute_error") / col("Revenue"))
mape = predictions.selectExpr("mean(percentage_error) as MAPE").collect()[0]["MAPE"] * 100

# Print the coefficients and intercept for linear regression
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

# Print the evaluation metrics
print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")
print(f"Mean Absolute Error (MAE) on test data: {mae}")
print(f"Mean Squared Error (MSE) on test data: {mse}")
print(f"Mean Absolute Percentage Error (MAPE) on test data: {mape}%")

# COMMAND ----------

# Show some sample predictions
predictions.select("prediction", "Revenue", "features").show(5)

# COMMAND ----------

# Calculate Confusion Matrix:

from pyspark.sql.functions import expr

predictions.groupBy("Revenue", "prediction").count().show()

# COMMAND ----------

# Save the trained Linear regression model
model_path = "./Internship_Sem-6_models/Ice_cream_Revenue_Prediction_model"
lr_model.save(model_path)

# COMMAND ----------

dbutils.fs.ls("dbfs:/Internship_Sem-6_models/Ice_cream_Revenue_Prediction_model")