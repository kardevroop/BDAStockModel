# -*- coding: utf-8 -*-
"""
Created on Tue May 14 18:02:06 2019

@author: user
"""

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import lit
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import IsotonicRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler
import pandas as pd
import numpy as np

# Convert u.data lines into (userID, movieID, rating) rows
def parseInput(line):
    fields = line.value.split()
    return Row(userID = int(fields[0]), movieID = int(fields[1]), rating = float(fields[2]))

def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])


if __name__ == "__main__":
    # Create a SparkSession
    spark = SparkSession.builder.appName("Stock Price Predictor").getOrCreate()

    # Load up our dataset
    #dataset = pd.read_csv('D:\\CM_BDA\\Data\\Daily\\daily_MSFT.csv')
    dataset = pd.read_csv('./Data/daily_MSFT.csv')
    #dataset = dataset.drop(columns = ['timestamp'])

    X = dataset.iloc[:,:]
    X = X.rename(columns = {'close':'label'})
    X = X.sort_values(by = 'timestamp')
    #X = spark.createDataFrame(X).cache()
    
    
    train_len = int(0.8 * len(X))
    test_len = len(X) - train_len
    X_train, X_test = X.iloc[0:train_len, :], X.iloc[train_len:len(X), :]

    # Convert to a DataFrame and cache it
    X_train = spark.createDataFrame(X_train).cache()
    X_test = spark.createDataFrame(X_test).cache()
    
        
    # Create an Random Forest Regression model from the complete data set
    data_cols = ['open','high','low','volume']
    assembler = VectorAssembler(inputCols=data_cols, outputCol="features")
    
    pipeline = Pipeline(stages=[assembler])
    #X = pipeline.fit(X).transform(X)
    
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

    # Compute summary statistics by fitting the StandardScaler
    #X = scaler.fit(X).transform(X)
    
    #X = transData(X)
    
    X_train = pipeline.fit(X_train).transform(X_train)
    X_test = pipeline.fit(X_test).transform(X_test)
    scalerModel = scaler.fit(X_train)
    X_train = scalerModel.transform(X_train)
    X_test = scalerModel.transform(X_test)
    '''
    X_train = transData(X_train)
    X_test = transData(X_test)
    '''
    #Splitting into train and test sets
    #X_train, X_test = X.randomSplit([0.8, 0.2])
    xtr = X_train.toPandas(); xtr.to_csv('X_train.csv')
    xtst = X_test.toPandas(); xtst.to_csv('X_test.csv')
    
    rf = RandomForestRegressor(labelCol = 'label', featuresCol="scaledFeatures", numTrees = 500)
    xgb = GBTRegressor(labelCol = 'label', featuresCol="features", maxIter = 100)
    glr = GeneralizedLinearRegression(family="gamma", link="inverse", maxIter=100, regParam=0.3)
    iso = IsotonicRegression()
    
    model = rf.fit(X_train)
    train_pred = model.transform(X_train)
    trpred = train_pred.toPandas(); trpred.to_csv('train_pred_rf.csv')
    
    pred = model.transform(X_test)
    tstpred = pred.toPandas(); tstpred.to_csv('test_pred_rf.csv')
        
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(pred)
    print("Root Mean Squared Error (RMSE) on test data for RF = %g" % rmse)
    
    model = xgb.fit(X_train)
    train_pred = model.transform(X_train)
    trpred = train_pred.toPandas(); trpred.to_csv('train_pred_gbt.csv')
    
    pred = model.transform(X_test)
    tstpred = pred.toPandas(); tstpred.to_csv('test_pred_gbt.csv')
        
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(pred)
    print("Root Mean Squared Error (RMSE) on test data for GBT = %g" % rmse)
    
    model = glr.fit(X_train)
    train_pred = model.transform(X_train)
    trpred = train_pred.toPandas(); trpred.to_csv('train_pred_gr.csv')
    
    pred = model.transform(X_test)
    tstpred = pred.toPandas(); tstpred.to_csv('test_pred_gr.csv')
        
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(pred)
    print("Root Mean Squared Error (RMSE) on test data for GLR = %g" % rmse)
    
    model = iso.fit(X_train)
    train_pred = model.transform(X_train)
    trpred = train_pred.toPandas(); trpred.to_csv('train_pred_iso.csv')
    
    pred = model.transform(X_test)
    tstpred = pred.toPandas(); tstpred.to_csv('test_pred_iso.csv')
        
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(pred)
    print("Root Mean Squared Error (RMSE) on test data for ISO = %g" % rmse)

    spark.stop()
