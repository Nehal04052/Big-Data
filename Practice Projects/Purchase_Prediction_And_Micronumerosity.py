# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Purchase Prediction & Effect of Micro-Numerosity
# MAGIC
# MAGIC #### **Description**:
# MAGIC
# MAGIC Customer Purchase Prediction involves leveraging machine learning algorithms to predict whether a customer will make a purchase based on various features such as age, gender, education, and review ratings. The Effect of Micro-Numerosity Model, in this context, refers to understanding how small variations in these features can influence purchasing behavior. By analyzing these attributes, machine learning models can identify patterns and correlations that might not be apparent through traditional analysis methods.
# MAGIC
# MAGIC #### **Key Features**:
# MAGIC
# MAGIC 1. **Age** : Different age groups may exhibit different purchasing behaviors. Younger customers might be more inclined towards trendy products, while older customers may prefer quality and reliability.
# MAGIC
# MAGIC 2. **Gender**: Gender-based preferences can significantly affect purchasing patterns. For instance, men and women might prioritize different aspects of a product.
# MAGIC
# MAGIC 3. **Education**: Education level can influence purchasing decisions, with more educated customers potentially focusing on the value and features of a product.
# MAGIC
# MAGIC 4. **Review**: Customer reviews play a crucial role in the decision-making process. Positive reviews can drive purchases, while negative reviews can deter potential buyers.
# MAGIC
# MAGIC 5. **Purchased**: Historical purchase data helps in understanding repeat buying patterns and customer loyalty.

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/tables/")

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler,StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

spark = SparkSession.builder.appName('Customer Purchase Prediction & Effect of Micro-Numerosity').getOrCreate() 

# COMMAND ----------

spark

# COMMAND ----------


df_pyspark = spark.read.csv('dbfs:/FileStore/tables/Customer_Purchase.csv',header=True,inferSchema=True)

# COMMAND ----------

df_pyspark.printSchema()

# COMMAND ----------

df_pyspark

# COMMAND ----------

df_pyspark.show()

# COMMAND ----------

# Drop any rows with null values (if any)
df_cleaned = df_pyspark.dropna()

# COMMAND ----------

# Convert categorical variables to numerical using StringIndexer
gender_indexer = StringIndexer(inputCol='Gender', outputCol='Gender_index')
education_indexer = StringIndexer(inputCol='Education', outputCol='Education_index')
review_indexer = StringIndexer(inputCol='Review', outputCol='Review_index')
purchased_indexer = StringIndexer(inputCol='Purchased', outputCol='Purchased_index')

df_indexed = gender_indexer.fit(df_cleaned).transform(df_cleaned)
df_indexed = education_indexer.fit(df_indexed).transform(df_indexed)
df_indexed = review_indexer.fit(df_indexed).transform(df_indexed)
df_indexed = purchased_indexer.fit(df_indexed).transform(df_indexed)

# COMMAND ----------

df_indexed.show()

# COMMAND ----------

# Select columns for RandomForestClassifier
feature_columns = ['Age', 'Gender_index', 'Education_index', 'Review_index']
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
df_assembled = assembler.transform(df_indexed)

# Convert target column to numeric
df_assembled = df_assembled.withColumn('label', col('Purchased_index'))

# Select final data for model training
data = df_assembled.select('features', 'label')

# Show schema of final prepared data
data.printSchema()

# COMMAND ----------

data.show()

# COMMAND ----------

# Split data into training and test sets
(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# Initialize RandomForestClassifier
rf = RandomForestClassifier(labelCol='label', featuresCol='features')

# Train model
model = rf.fit(train_data)

# COMMAND ----------

# Make predictions
predictions = model.transform(test_data)

# Show predictions
predictions.select('label', 'prediction', 'probability').show(10, False)

# COMMAND ----------

# Confusion matrix
predictions.groupBy('label', 'prediction').count().show()

# Evaluate model using accuracy, precision, recall, and F1-score
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction')

# Accuracy
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
print(f"Accuracy: {accuracy}")

# Precision
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
print(f"Precision: {precision}")

# Recall
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
print(f"Recall: {recall}")

# F1-score
f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print(f"F1-Score: {f1_score}")


# COMMAND ----------

# Save the trained logistic regression model
model_path = "./Internship_Sem-6_models/Customer_Purchase_Prediction_model"
model.save(model_path)

# COMMAND ----------

dbutils.fs.ls("dbfs:/Internship_Sem-6_models/Customer_Purchase_Prediction_model")