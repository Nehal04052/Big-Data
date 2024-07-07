# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Card Default Prediction
# MAGIC ####The data set consists of 2000 samples from each of two categories. Five variables are
# MAGIC
# MAGIC 1. Income
# MAGIC 2. Age
# MAGIC 3. Loan
# MAGIC 4. Loan to Income (engineered feature)
# MAGIC 5. Default

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/tables/")

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

# COMMAND ----------

spark = SparkSession.builder.appName('Credit Card Default Prediction').getOrCreate() 

# COMMAND ----------

spark

# COMMAND ----------

df_pyspark = spark.read.csv('dbfs:/FileStore/tables/Credit_Default.csv',header=True,inferSchema=True)

# COMMAND ----------

df_pyspark.printSchema()

# COMMAND ----------

df_pyspark

# COMMAND ----------

df_pyspark.show()

# COMMAND ----------

# Drop rows with null values if necessary
df_pyspark= df_pyspark.dropna()

# COMMAND ----------

# Assemble features into a vector
feature_cols = df_pyspark.columns[:4]  # Excluding 'diagnosis'
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df_pyspark)

# COMMAND ----------

df_assembled.select("features","Default").show()

# COMMAND ----------

# Split data into training and testing sets
train_data, test_data = df_assembled.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# Initialize logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="Default")

# Fit the model
lr_model = lr.fit(train_data)

# COMMAND ----------

# Make predictions
predictions = lr_model.transform(test_data)

# Evaluate the model using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol="Default")
accuracy_binary = evaluator.evaluate(predictions)

# Compute confusion matrix
confusion_matrix = predictions.groupBy('Default').pivot('prediction').count().na.fill(0).orderBy('Default')
confusion_matrix.show()

# Compute classification report (precision, recall, f1-score)
tp = predictions.filter((col("Default") == 1.0) & (col("prediction") == 1.0)).count()
tn = predictions.filter((col("Default") == 0.0) & (col("prediction") == 0.0)).count()
fp = predictions.filter((col("Default") == 0.0) & (col("prediction") == 1.0)).count()
fn = predictions.filter((col("Default") == 1.0) & (col("prediction") == 0.0)).count()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Binary Classification Accuracy: {accuracy_binary}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1_score}")

# COMMAND ----------

# Save the trained logistic regression model
model_path = "./Internship_Sem-6_models/Credit_Card_Default_Prediction_model"
lr_model.save(model_path)

# COMMAND ----------

dbutils.fs.ls("dbfs:/Internship_Sem-6_models/Credit_Card_Default_Prediction_model")