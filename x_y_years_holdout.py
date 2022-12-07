# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### X Y years Holdout
# MAGIC This notebook takes the diagnosis table data and split it between the most recent year to be used as the hold out data and the other two years which are used as the training data.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Remove Redundant Columns, convert to Pandas and Seperate Y

# COMMAND ----------

import pandas as pd

def one_hot_encode_column(column_name : str, dataframe : pd.DataFrame) -> pd.DataFrame:
  """
  One hot encoding for column specified by column
  """
  return dataframe.join(pd.get_dummies(dataframe[column_name], prefix=f"{column_name}")).drop(columns=[column_name])

# COMMAND ----------

def create_x_y(dataset):
  """
  This function does the splitting into a feature (x) and target (y) sets, as well as one hot encoding certain features
  Features in the one hot encode list are one hot encoded.
  """ 
  X = dataset
  
  X = X[X.columns].apply(pd.to_numeric, errors='coerce')
  
  Y = X.pop("DIED").astype('int8')
    
  return X, Y

def create_holdout(SHMI_data, diag_group:int):
  """
  This function creates a holdout dataset by using the most recent year as the holdout and training on the previous years.
  
  Hold out year is the most recent year.
  
  This functions takes SHMI_data as a pyspark dataframe and diag_group as an interger.
  It then drops uneeded columns and seperates the data into X and Y, with the output being
  X and Y as pandas dataframes as well as the training data X_train and Y_train and Holdout data
  X_test and Y_test.
  Returns: X, Y, X_train, X_test, Y_train, Y_test ; all pandas dataframes
  """
  drop_cols = ['SUSSPELLID','EPIORDER', 'SHMI_DIAGNOSIS_GROUP'] #+ columns # Drop these as they are irrelevant to the model.
  
  SHMI_one_diag_group_pdf = SHMI_data.where(f"SHMI_DIAGNOSIS_GROUP == {diag_group}").drop(*drop_cols).toPandas()
  
  # apply the one hold encoding to ADMIETH, ETH_Group, GENDER - Eth Group removed as it wasn't signed off
  one_hot_encode_list=['ADMIMETH', 'GENDER']
  for column_name in one_hot_encode_list:
    SHMI_one_diag_group_pdf = one_hot_encode_column(column_name, SHMI_one_diag_group_pdf)
  
  X, Y = create_x_y(SHMI_one_diag_group_pdf)
  
  ## test set
  test = SHMI_one_diag_group_pdf[SHMI_one_diag_group_pdf['YEAR_INDEX'] == 1]
  
  X_test, Y_test = create_x_y(test) 
  
  
  ## train set
  train = SHMI_one_diag_group_pdf[SHMI_one_diag_group_pdf['YEAR_INDEX'] != 1]
  
  X_train, Y_train = create_x_y(train)
  
  #make them sparse
  X_test = data_frame_to_scipy_sparse_matrix(X_test)
  X_train = data_frame_to_scipy_sparse_matrix(X_train)
  
  return X, Y, X_train, X_test, Y_train, Y_test
