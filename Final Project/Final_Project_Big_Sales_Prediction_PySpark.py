# Databricks notebook source
# MAGIC %md
# MAGIC # **Big Sales Prediction using Random Forest Regressor**

# COMMAND ----------

# MAGIC %md
# MAGIC -------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Objective**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Predict the sales of items in different stores using a Random Forest Regressor.
# MAGIC
# MAGIC There are 12 variables in dataset.
# MAGIC
# MAGIC 1. Item_Identifier
# MAGIC 2. Item_Weight: 
# MAGIC 3. Item_Fat_Content
# MAGIC 4. Item_Visibility
# MAGIC 5. Item_Type
# MAGIC 6. Item_MRP
# MAGIC 7. Outlet_Identifier
# MAGIC 8. Outlet_Establishment_Year
# MAGIC 9. Outlet_Size
# MAGIC 10. Outlet_Location_Type
# MAGIC 11. Outlet_Type
# MAGIC 12. Item_Outlet_Sales

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Data Source**

# COMMAND ----------

# MAGIC %md
# MAGIC ####**Big Sales Data CSV File Link** :- https://github.com/YBIFoundation/Dataset/blob/main/Big%20Sales%20Data.csv

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/tables/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Import Library**

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Spark Session

# COMMAND ----------

spark = SparkSession.builder.appName("BigSalesPrediction").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Import Data**

# COMMAND ----------

df_pyspark = spark.read.csv("dbfs:/FileStore/tables/Big_Sales_Data.csv", header=True, inferSchema=True)

# COMMAND ----------

df_pyspark.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Describe Data**

# COMMAND ----------

df_pyspark.describe().show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## **Data Visualization**

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = df_pyspark.toPandas()
df= df.dropna()
sns.pairplot(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Data Preprocessing**

# COMMAND ----------

df[['Item_Identifier']].value_counts()

# COMMAND ----------

df[['Item_Fat_Content']].value_counts()

# COMMAND ----------

df.replace({'Item_Fat_Content':{'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'}},inplace=True)

# COMMAND ----------

df[['Item_Fat_Content']].value_counts()

# COMMAND ----------

df.replace({'Item_Fat_Content': {'Low Fat':0,'Regular':1}},inplace=True)

# COMMAND ----------

df[['Item_Type']].value_counts()

# COMMAND ----------

df.replace({'Item_Type':{
    'Fruits and Vegetables':0,'Snack Foods':0,'Household':1,
    'Frozen Foods':0,'Dairy':0,'Baking Goods':0,
    'Canned':0,'Health and Hygiene':1,
    'Meat':0,'Soft Drinks':0,'Breads':0,'Hard Drinks':0,
    'Others':2,'Starchy Foods':0,'Breakfast':0,'Seafood':0
}},inplace=True)

# COMMAND ----------

df[['Item_Type']].value_counts()

# COMMAND ----------

df[['Outlet_Identifier']].value_counts()

# COMMAND ----------

df.replace({'Outlet_Identifier':{'OUT027':0,'OUT013':1,
                                'OUT049':2,'OUT046':3,'OUT035':4,
                                'OUT045':5,'OUT018':6,
                                'OUT017':7,'OUT010':8,'OUT019':9
                                }},inplace=True)

# COMMAND ----------

df[['Outlet_Identifier']].value_counts()

# COMMAND ----------

df[['Outlet_Size']].value_counts()

# COMMAND ----------

df.replace({'Outlet_Size':{'Small':0,'Medium':1,'High':2}},inplace=True)

# COMMAND ----------

df[['Outlet_Size']].value_counts()

# COMMAND ----------

df[['Outlet_Location_Type']].value_counts()

# COMMAND ----------

df.replace({'Outlet_Location_Type':{'Tier 1':0,'Tier 2':1,'Tier 3':2}},inplace=True)

# COMMAND ----------

df[['Outlet_Location_Type']].value_counts()

# COMMAND ----------

df[['Outlet_Type']].value_counts()

# COMMAND ----------

df.replace({'Outlet_Type':{'Grocery Store':0,'Supermarket Type1':1,'Supermarket Type2':2,'Supermarket Type3':3}},inplace=True)

# COMMAND ----------

df[['Outlet_Type']].value_counts()

# COMMAND ----------

df.head()

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Define Target Variable (y) and Feature Variables (X)**

# COMMAND ----------

y= df['Item_Outlet_Sales']


# COMMAND ----------

y

# COMMAND ----------

X = df[[
        'Item_Weight','Item_Fat_Content','Item_Visibility',
        'Item_Type','Item_MRP','Outlet_Identifier',
        'Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type',
        'Outlet_Type'
]]

# COMMAND ----------

X

# COMMAND ----------

# MAGIC %md
# MAGIC #Get X Variable Standardized

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

# COMMAND ----------

sc = StandardScaler()

# COMMAND ----------

X_std = df[['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']]

# COMMAND ----------

X_std = sc.fit_transform(X_std)

# COMMAND ----------

X_std

# COMMAND ----------

X[['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']] = pd.DataFrame(X_std, columns= [['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']])

# COMMAND ----------

X

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Train Test Split**

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=12529)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Modeling**

# COMMAND ----------

# Fill or drop null values
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
y_train.fillna(0, inplace=True)
y_test.fillna(0, inplace=True)

# Concatenate X and y for both training and testing datasets
train_pd = pd.concat([X_train, y_train], axis=1)
test_pd = pd.concat([X_test, y_test], axis=1)

# Convert Pandas DataFrames to PySpark DataFrames
train_spark = spark.createDataFrame(train_pd)
test_spark = spark.createDataFrame(test_pd)

# Assuming feature columns were already defined in previous preprocessing steps
feature_cols = X_train.columns.tolist()  # List of feature columns
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Transform training and testing datasets
train_spark = assembler.transform(train_spark).select("features", 'Item_Outlet_Sales')
test_spark = assembler.transform(test_spark).select("features", 'Item_Outlet_Sales')

# Modeling
rf = RandomForestRegressor(featuresCol="features", labelCol='Item_Outlet_Sales')  # Adjust maxBins as needed
rf_model = rf.fit(train_spark)


# COMMAND ----------

# MAGIC %md
# MAGIC ## **Model Evaluation**

# COMMAND ----------

# Model Evaluation
predictions = rf_model.transform(test_spark)
evaluator_rmse = RegressionEvaluator(labelCol="Item_Outlet_Sales", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="Item_Outlet_Sales", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="Item_Outlet_Sales", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")
print(f"Mean Absolute Error (MAE) on test data = {mae}")
print(f"R² Score on test data = {r2}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Visualization of Actual Vs Predict Results

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Convert predictions to Pandas DataFrame for plotting
predictions_pd = predictions.select("Item_Outlet_Sales", "prediction").toPandas()

# Extract actual and predicted values
actual_values = predictions_pd["Item_Outlet_Sales"]
predicted_values = predictions_pd["prediction"]

# Plotting Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(actual_values, predicted_values, alpha=0.5)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Explaination**

# COMMAND ----------

# MAGIC %md
# MAGIC The Random Forest Regressor model was trained on the given dataset to predict the sales of items in different stores. The model evaluation shows how well the model performs on unseen data by calculating the Root Mean Squared Error (RMSE). The same preprocessing steps must be applied to any new data before making predictions.