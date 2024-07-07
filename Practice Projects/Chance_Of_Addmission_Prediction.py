# Databricks notebook source
# MAGIC %md
# MAGIC # Chance of Admission for Higher Studies
# MAGIC #### Predict the chances of admission of a student to a Graduate program based on:
# MAGIC
# MAGIC 1. GRE Scores (290 to 340)
# MAGIC 2. TOEFL Scores (92 to 120)
# MAGIC 3. University Rating (1 to 5)
# MAGIC 4. Statement of Purpose (1 to 5)
# MAGIC 5. Letter of Recommendation Strength (1 to 5)
# MAGIC 6. Undergraduate CGPA (6.8 to 9.92)
# MAGIC 7. Research Experience (0 or 1)
# MAGIC 8. Chance of Admit (0.34 to 0.97)

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/tables/")

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, abs
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

spark = SparkSession.builder.appName('Chance of Admission for Higher Studies').getOrCreate() 

# COMMAND ----------

spark

# COMMAND ----------


df_pyspark = spark.read.csv('dbfs:/FileStore/tables/Admission_Chance.csv',header=True,inferSchema=True)

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

# Clean column names by trimming spaces
for col_name in df_pyspark.columns:
    df_pyspark = df_pyspark.withColumnRenamed(col_name, col_name.strip().replace(' ', '_'))

# Show schema after cleaning
df_pyspark.printSchema()

# COMMAND ----------

# Handle missing values if necessary
df_pyspark = df_pyspark.na.drop()

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare the DataFrame

# COMMAND ----------

# Define the feature columns
feature_columns = ["GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR", "CGPA", "Research"]

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_pyspark = assembler.transform(df_pyspark)

# Select only the features and target column
df_pyspark = df_pyspark.select("features", "Chance_of_Admit")

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
lr = LinearRegression(featuresCol="features", labelCol="Chance_of_Admit")

# Fit the model on the training data
lr_model = lr.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Evaluate the Model

# COMMAND ----------

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Evaluate the model using RMSE
rmse_evaluator = RegressionEvaluator(labelCol="Chance_of_Admit", predictionCol="prediction", metricName="rmse")
rmse = rmse_evaluator.evaluate(predictions)

# Evaluate the model using MAE
mae_evaluator = RegressionEvaluator(labelCol="Chance_of_Admit", predictionCol="prediction", metricName="mae")
mae = mae_evaluator.evaluate(predictions)

# Evaluate the model using MSE
mse_evaluator = RegressionEvaluator(labelCol="Chance_of_Admit", predictionCol="prediction", metricName="mse")
mse = mse_evaluator.evaluate(predictions)

# Calculate MAPE (Mean Absolute Percentage Error)
predictions = predictions.withColumn("absolute_error", abs(col("prediction") - col("Chance_of_Admit")))
predictions = predictions.withColumn("percentage_error", col("absolute_error") / col("Chance_of_Admit"))
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
predictions.select("prediction", "Chance_of_Admit", "features").show(5)

# COMMAND ----------

# Calculate Confusion Matrix:

from pyspark.sql.functions import expr

predictions.groupBy("Chance_of_Admit", "prediction").count().show()


# COMMAND ----------

# Save the trained linear regression model
model_path = "./Internship_Sem-6_models/Chance_of_addmission_model"
lr_model.save(model_path)

# COMMAND ----------

dbutils.fs.ls("dbfs:/Internship_Sem-6_models/Chance_of_addmission_model")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /tmp

# COMMAND ----------

dbutils.fs.cp("dbfs:/Chance_of_addmission_model", "file:/tmp/Chance_of_addmission_model",recurse=True)


# COMMAND ----------

# MAGIC %sh
# MAGIC ls /tmp/Chance_of_addmission_model

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC zip -r /tmp/Chance_of_addmission_model.zip /tmp/Chance_of_addmission_model
# MAGIC

# COMMAND ----------

dbutils.fs.cp("file:/tmp/Chance_of_addmission_model.zip","dbfs:/FileStore/Chance_of_addmission_model.zip")


# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore")
#dbutils.fs.rm("dbfs:/Chance_of_addmission_model/",recurse=True)