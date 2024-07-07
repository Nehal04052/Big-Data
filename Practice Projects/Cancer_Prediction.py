# Databricks notebook source
# MAGIC %md
# MAGIC # Cancer Prediction
# MAGIC
# MAGIC ## Dataset Information:
# MAGIC
# MAGIC ####Target Variable (y):
# MAGIC
# MAGIC - Diagnosis (M = malignant, B = benign)
# MAGIC
# MAGIC #### Ten features (X) are computed for each cell nucleus:
# MAGIC
# MAGIC 1. radius (mean of distances from center to points on the perimeter)
# MAGIC 2. texture (standard deviation of gray-scale values)
# MAGIC 3. perimeter
# MAGIC 4. area
# MAGIC 5. smoothness (local variation in radius lengths)
# MAGIC 6. compactness (perimeter^2 / area - 1.0)
# MAGIC 7. concavity (severity of concave portions of the contour)
# MAGIC 8. concave points (number of concave portions of the contour)
# MAGIC 9. symmetry
# MAGIC 10. fractal dimension (coastline approximation - 1)
# MAGIC
# MAGIC #### For each characteristic three measures are given:
# MAGIC
# MAGIC    a. Mean
# MAGIC
# MAGIC    b. Standard error
# MAGIC
# MAGIC    c. Largest/ Worst

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

spark = SparkSession.builder.appName('Cancer Prediction').getOrCreate() 

# COMMAND ----------

spark

# COMMAND ----------

df_pyspark = spark.read.csv('dbfs:/FileStore/tables/Cancer.csv',header=True,inferSchema=True)

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

# Convert diagnosis column to numeric (0 for benign, 1 for malignant)
indexer = StringIndexer(inputCol="diagnosis", outputCol="label")
indexed_data = indexer.fit(df_pyspark).transform(df_pyspark)
indexed_data = indexed_data.withColumn("label", indexed_data["label"].cast("integer"))

# COMMAND ----------

indexed_data.select("diagnosis","label").show()

# COMMAND ----------

# Clean dataframe (keep all columns except id and _c32)
columns_to_keep = [
    "label",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

df_cleaned = indexed_data.select(columns_to_keep)

# COMMAND ----------

df_cleaned.show()

# COMMAND ----------

# Drop rows with null values if necessary
df_cleaned = df_cleaned.dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare the DataFrame

# COMMAND ----------

# Assemble features into a vector
feature_cols = df_cleaned.columns[1:]  # Excluding 'diagnosis'
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df_cleaned)

# COMMAND ----------

df_assembled.select("features","label").show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Split the DataFrame

# COMMAND ----------

# Split data into training and testing sets
train_data, test_data = df_assembled.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Train the Model

# COMMAND ----------

# Initialize logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Fit the model
lr_model = lr.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Evaluate the Model

# COMMAND ----------

# Make predictions
predictions = lr_model.transform(test_data)

# Evaluate the model using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol="label")
accuracy_binary = evaluator.evaluate(predictions)

# Compute confusion matrix
confusion_matrix = predictions.groupBy('label').pivot('prediction').count().na.fill(0).orderBy('label')
confusion_matrix.show()

# Compute classification report (precision, recall, f1-score)
tp = predictions.filter((col("label") == 1.0) & (col("prediction") == 1.0)).count()
tn = predictions.filter((col("label") == 0.0) & (col("prediction") == 0.0)).count()
fp = predictions.filter((col("label") == 0.0) & (col("prediction") == 1.0)).count()
fn = predictions.filter((col("label") == 1.0) & (col("prediction") == 0.0)).count()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Binary Classification Accuracy: {accuracy_binary}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1_score}")

# COMMAND ----------

# Save the trained logistic regression model
model_path = "./Internship_Sem-6_models/Cancer_Prediction_model"
lr_model.save(model_path)

# COMMAND ----------

dbutils.fs.ls("dbfs:/Internship_Sem-6_models/Cancer_Prediction_model")