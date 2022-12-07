# Databricks notebook source
# MAGIC %md
# MAGIC # Run model function
# MAGIC This notebook is used to run models at the same time as other models in other notebooks

# COMMAND ----------

# DBTITLE 1,Sklearn Version
import sklearn
print("Sklearn version: ")
sklearn.__version__

# COMMAND ----------

import xgboost
print("xgboost version: ")
xgboost.__version__

# COMMAND ----------

import statsmodels
print("statsmodel version: ")
statsmodels.__version__

# COMMAND ----------

# DBTITLE 1,Imports

# Try stats models for Logisitcgressions

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import recall_score, f1_score, make_scorer, auc, average_precision_score, classification_report,confusion_matrix, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn import svm

import xgboost as xgb

  
scoring = {"prec_recall_auc" : 'average_precision', "precision" : 'precision_macro', "recall" : 'recall_macro', "roc_auc" : 'roc_auc'} # [ptentially create own scoring function whcih actually works]. it can take function names

# COMMAND ----------

# DBTITLE 1,List of Models
list_of_models = {
 'LogisticRegression_sag_grid' : {"model": LogisticRegression(solver= 'sag', max_iter=1000), "grid_search" : True, "params" : {'model__C' : [10, 1.0, 0.1, 0.01]}}, #'model__max_iter' : [100,250,500,750,1000]}},
 'LogisticRegression_l1_grid' : {"model": LogisticRegression(penalty='l1', solver= 'saga', max_iter=1000), "grid_search" : True, "params" : {'model__C' : [10, 1.0, 0.1, 0.01]}}, #'model__max_iter' : [100,250,500,750,1000]}},
  
 
 'LogisticRegression_sag_c10' : {"model": LogisticRegression( C= 10, solver= 'sag', max_iter=1000), "grid_search" : False}, #changed max iterations to 1000. Previously it was default 100
 'Calib_LogisticRegression_sag_c10' : {"model": CalibratedClassifierCV(LogisticRegression( C= 10, solver= 'sag', max_iter=1000), cv=5), "grid_search" : False},
 'Calib_LogReg_500_sag_c10' : {"model": CalibratedClassifierCV(LogisticRegression( C= 10, solver= 'sag', max_iter=500), cv=5), "grid_search" : False},

 'Calib_LogisticRegression_sag' : {"model": CalibratedClassifierCV(LogisticRegression(solver= 'sag', max_iter=1000), cv=5), "grid_search" : False}, #default c value
  
  'LogisticRegression_l1_saga' : {"model": LogisticRegression( penalty='l1', C= 10, solver= 'saga', max_iter=1000), "grid_search" : False},
 'Calib_LogisticRegression_l1_saga' : {"model": CalibratedClassifierCV(LogisticRegression( penalty='l1', C= 10, solver= 'saga', max_iter=1000), cv=5), "grid_search" : False},
 'Calib_LogReg_500_l1_saga' : {"model": CalibratedClassifierCV(LogisticRegression( penalty='l1', C= 10, solver= 'saga', max_iter=500), cv=5), "grid_search" : False},
  
  'Calib_LogisticRegression_l1' : {"model": CalibratedClassifierCV(LogisticRegression( penalty='l1', solver= 'saga', max_iter=1000), cv=5), "grid_search" : False}, #default c value
  
  'poly2_logreg_sag_c10' : {"model": [('polynomial_features',PolynomialFeatures(degree = 2, interaction_only=False, include_bias=False)), ("model", LogisticRegression( C= 10, solver= 'sag', max_iter=1000))],"grid_search" : False},
  
 'Calib_XGBoost_default' : {"model": CalibratedClassifierCV(xgb.XGBClassifier(tree_method='hist'), cv=5), "grid_search" : False},
 #'XGBoost_1' : {"model": xgb.XGBClassifier(tree_method='hist', max_delta_step=1, max_depth=48, n_estimators=500), "grid_search" : False},
 'XGBoost_1' : {"model": xgb.XGBClassifier(tree_method='hist', max_depth=48, n_estimators=500), "grid_search" : False},
 'XGBoost_deeper_grid' : {"model": xgb.XGBClassifier(tree_method='hist'), "grid_search" : True, "params" :  { # aims to help with underfitting
   # 'pca__n_components': [5, 10, 15, 20, 25, 30],
    'model__max_depth': [4,5,6,7,8,9,10], # reason: Default is 6. Maybe not deep enough and so not learning enough, so make it deeper. Do not want to overfit so no more than 10. Origional model may already be overfit, so trying below 6 as well.
    'model__learning_rate':[0.1, 0.2, 0.3], # Reason: default is 0.3. May be learning too fast and so not reaching the optimum. Start at a lower one and gradually improve
    'model__n_estimators': [100,500,1000], # reason: default is 100. 
    'model__colsample_bytree' : [0.5, 1] # reason:  Default is 1. This can help improve overfitting. sampling less may make it generalise more.
    #'model__subsample' : [0.6,0.7,0.8,0.9, 1] # this is used to help with overfitting. Its the subsample of observations used. But too low and it will lead to underfitting. 1 is default/
    #'model__max_delta_step': [0,1,2]
}},
  'XGBoost_grid_2' : {"model": xgb.XGBClassifier(tree_method='hist'), "grid_search" : True, "params" :  {
   # 'pca__n_components': [5, 10, 15, 20, 25, 30], # reason:
    'model__max_depth': [3,4,5,6,7,8,9,10], # reason: Default is 6. Maybe not deep enough and so not learning enough, so make it deeper. Do not want to overfit so no more than 10. Origional model may already be overfit, so trying below 6 as well.
    #'model__learning_rate':[0.1, 0.2, 0.3], # Reason: default is 0.3. May be learning too fast and so not reaching the optimum. Start at a lower one and gradually improve
    #'model__n_estimators': [100,500,1000], # reason: default is 100. 
    'model__colsample_bytree' : [0.5, 1], # reason:  Default is 1. This can help improve overfitting. sampling less may make it generalise more.
    'model__subsample' : [0.6,0.7,0.8,0.9, 1], # this is used to help with overfitting. Its the subsample of observations used. But too low and it will lead to underfitting. 1 is default/
    'model__gamma': [0,1,2,3], # can help with overfitting, by making it harder to created new leafs - Xgboost documentation
    'model__min_child_weight' : [0,1,2,3], # can help with overfitting, by making it harder to created new leafs - Xgboost documentation
    'model__max_delta_step': [0,1,2,3] # can help with imbalanced datasets - Xgboost documentation
}},
  'XGboost_optimal22_1st' : {"model": xgb.XGBClassifier(tree_method='hist',colsample_bytree= 0.5, learning_rate= 0.1, max_depth= 10, n_estimators= 100), "grid_search" : False}, # First from grid with 0.4 PR AUC on diag22. 
  'XGboost_optimal22_3rd' : {"model": xgb.XGBClassifier(tree_method='hist',colsample_bytree= 0.5, learning_rate= 0.1, max_depth= 4, n_estimators= 100), "grid_search" : False}, # This was the 3rd best after the grid search, but almost twice as fast as the first with little drop in fit stat (micrscropic)
 # diag 39 {'model__colsample_bytree': 0.5, 'model__learning_rate': 0.1, 'model__max_depth': 10, 'model__n_estimators': 1000}
  'XGboost_dg39_deep10' : {"model": xgb.XGBClassifier(tree_method='hist', max_depth= 10), "grid_search" : False}, # See above for the best grid search for this. Saved the output to a notebook in the CCS folder
  'XGboost_dg39_deep15' : {"model": xgb.XGBClassifier(tree_method='hist', max_depth= 15), "grid_search" : False},
  'XGboost_dg39_deep20' : {"model": xgb.XGBClassifier(tree_method='hist', max_depth= 20), "grid_search" : False},
  'XGboost_dg39_deep25' : {"model": xgb.XGBClassifier(tree_method='hist', max_depth= 25), "grid_search" : False},
  'XGboost_dg39_deep30' : {"model": xgb.XGBClassifier(tree_method='hist', max_depth= 30), "grid_search" : False},
  
  
 'DummyClassifier' : {"model" : DummyClassifier(strategy="stratified"), "grid_search" : False},
 'Calib_Dummy' : {"model" : CalibratedClassifierCV(DummyClassifier(strategy="stratified"), cv=5), "grid_search" : False}, 
  
 'Calibrated_Dec_Tree_0' : {"model" : CalibratedClassifierCV(DecisionTreeClassifier(class_weight="balanced"), method='sigmoid', cv=5), "grid_search" : False},
 'Dec_Tree_0' : {"model" : DecisionTreeClassifier(class_weight="balanced"), "grid_search" : False},
  
 'SVM_0' : {"model" : svm.SVC(class_weight='balanced'), "grid_search" : False} # balanced class weight, others default
}

# COMMAND ----------

# DBTITLE 1,run_model_score function
def run_model_score(model,grid_search=False, params=None, scoring=scoring):
  """
  This function creates a pipeline with the selected model and outputs a scores
  Args: model is an Sklearn model e.g. LogisticRegression()
  params is a place holder variable that might be used to store params for grid search
  output: model scores as a pandas dataframe. Currently: precision, recal and roc
  """
  if 'poly' in model_name:
    pipe = Pipeline(model)
  else:
    pipe = Pipeline([('model', model)]) #penalty = 'l2', solver='saga', # Age, IMD score, if spline or polynominal dont need to standarise. Maybe standardise
    
  #n_samples = X_train.shape[0]
  cv = StratifiedKFold (n_splits=5,random_state=0) #don't use the shuffle one as it will cause overlap in the test data.

  if grid_search == True:
    search = GridSearchCV(pipe, params, scoring='average_precision', cv=cv).fit(X_train, Y_train)
    return search.cv_results_, search.best_params_, search.best_score_

  else:
    pipe.fit(X_train,Y_train)
    scores = cross_validate(pipe, X_train, Y_train, cv=cv, scoring=scoring, return_train_score=True)
    Y_pred = pipe.predict(X_test)
    Y_pred_proba = pipe.predict_proba(X_test)[:,1]
    
    #return pd.DataFrame(scores) ,pd.DataFrame(pd.DataFrame(scores).mean()).T, confusion_matrix(Y_test, Y_pred), Y_pred, Y_pred_proba, pipe
    return pd.DataFrame(scores), Y_pred, Y_pred_proba, pipe

# COMMAND ----------

# MAGIC %md
# MAGIC ### def pandas_score_output(model_name, Y_test = Y_test, Y_pred_proba = Y_pred_proba)

# COMMAND ----------

import numpy as np
import pandas as pd

def pandas_score_output(model_name : str, Y_test: np.ndarray, Y_pred_proba: np.ndarray) -> pd.DataFrame:
  """
  This take a target pandas dataframe (Y_test) and a prediction probability pandas dataframe (Y_pred_proba) and outputs a pandas dataframe containing the following metrics:
  "PR_AUC_SCORE": 
  ROC_AUC_SCORE": 
   "f_score_live" 
  "precision_live"
  "recall_live" : 
  "support_live" :
  """
  from sklearn.metrics import recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, precision_recall_fscore_support
  precision, recall, f_score, support = precision_recall_fscore_support(Y_test, Y_pred_proba.round())
  
  #model_name = "DummyClassifier"
  return pd.DataFrame({'Model_Name': [model_name],
                "PR_AUC_SCORE": [average_precision_score(Y_test, Y_pred_proba)],
               "ROC_AUC_SCORE": [roc_auc_score(Y_test, Y_pred_proba)],
                 "f_score_live" : [f_score[0]],  "f_score_dead" : [f_score[1]],
                "precision_live" : [precision[0]], "precision_dead" : [precision[1]],
                "recall_live" : [recall[0]] , "recall_dead" : [recall[1]], 
                "support_live" : [support[0]] , "support_dead" : [support[1]]
               }).round(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### plot_precision_recall_curve(Y_test = Y_test ,Y_pred = Y_pred)

# COMMAND ----------

def plot_precision_recall_curve(Y_test: np.ndarray, Y_pred: np.ndarray):
  """
  This function plots a PR Curve taking the target values Y and the predicted target values Y_pred as input. It outputs a PR curve plot
  Defaults to the test set
  """
  
  precision, recall, threshold = precision_recall_curve(Y_test, Y_pred)
  
  
  #create precision recall curve
  fig, ax = plt.subplots()
  ax.plot(recall, precision, color='purple')
  
  #add axis labels to plot
  ax.set_title('Precision-Recall Curve')
  ax.set_ylabel('Precision')
  ax.set_xlabel('Recall')
  
  #display plot
  return display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## show_coefficients(fit_model, X)
# MAGIC - if its a pipe then fit_model should be fit_model.named_steps['model']

# COMMAND ----------

def show_coefficients(fit_model, X):
  """
  This functions take the X data as a pandas dataframe as X, and the fitted model pipeline as fit_model and outputs a pandas dataframe with the coeffiicents for each column, sorted by value. It also prints the 
  """
  
  if "Calib" in model_name:
    coef_avg = 0
    intercept_avg = 0
    for i in fit_model.calibrated_classifiers_:
      coef_avg = coef_avg + i.base_estimator.coef_
      intercept_avg = intercept_avg + i.base_estimator.intercept_
    coef  = coef_avg/len(fit_model.calibrated_classifiers_)
    intercept  = intercept_avg/len(fit_model.calibrated_classifiers_)   
    
  else:
    coef = fit_model.coef_[0, :]
    intercept = fit_model.intercept_
    coef
    print("The intercept for the mode:", intercept)
    print("The coefficients for the model, sorted by coef")
    return pd.DataFrame(coef, 
               X.columns, 
               columns=['coef'])\
              .sort_values(by='coef', ascending=False)


# COMMAND ----------

# DBTITLE 1,Extract Calibration data

from pyspark.sql import DataFrame as DataFrame
import numpy as np

def extract_calibration_data(saved_calibration_data : DataFrame, model_name, chosen_diag_group) -> (np.array, np.array):
  """
  This functiont takes a pyspark dataframe filled with calibration plot data (fraction_of_positives, mean_predicted_value)
  for each model and diagnosis group, and extracts it as two np.arrays called fraction_of_positives, mean_predicted_value.
  This allows them to be used to create calibration plots
  """
  data = saved_calibration_data.where(f"model ='{model_name}' and diag_group = '{chosen_diag_group}'")

  fraction_of_positives = np.array(np.float_(data.first()['fraction_of_positives'][1:][:-1].split()))
  mean_predicted_value = np.array(np.float_(data.first()['mean_predicted_value'][1:][:-1].split()))
  return fraction_of_positives, mean_predicted_value



# COMMAND ----------

# DBTITLE 1,Calibration code
from pyspark.sql.functions import col, lit, concat
import matplotlib
from cycler import cycler

def plot_calibration_curve(chosen_diag_group: int, plot_duplicate_models=False, all_models=False, chosen_palette='tab20') -> None:
  """
  This function plots a calibration curve of the models in a diagnosis group. If all_models is true then all models are plotted, if flase then only one of each model is plotted. 
  If plot duplicates is fasle then only the highest scoring models are plotted
  """
  if all_models==True:
    saved_calibration_data = spark.sql(f"""select * FROM {dboutput}.{calibration_table}""").where(f'diag_group = {chosen_diag_group}')
  elif plot_duplicate_models==False:
    saved_calibration_data = spark.sql(f"""select * 
                FROM 
                (
                SELECT *,
              dense_rank() OVER (PARTITION BY Model,Diag_Group ORDER BY PR_AUC_SCORE DESC) AS rank
          FROM {dboutput}.{calibration_table}
      ) vo WHERE rank = 1
       ORDER BY Diag_Group
                """).drop('rank').dropDuplicates(['Model','Diag_Group']).where(f'diag_group = {chosen_diag_group}')
  else: 
    saved_calibration_data = spark.sql(f"""select * FROM {dboutput}.{calibration_table}""").where(f'diag_group = {chosen_diag_group}')
    saved_calibration_data = saved_calibration_data.withColumn("model", concat(col("model"), lit("_PR_"), col("PR_AUC_SCORE")))
  
  models_list = saved_calibration_data.toPandas().sort_values("model")['model'].drop_duplicates().tolist() #get a list of the models that have been run
  
  fig = plt.figure(figsize=(8.5, 5))
  
  plt.rcParams['axes.prop_cycle'] = cycler('color', plt.get_cmap(chosen_palette).colors)
  
  #set up the axes
  ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
  
  #plot the perfect example
  ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
  
  for model_name in models_list:
    # extract the calibration data so its in the same format as it would be from the SK learn thing
    fraction_of_positives, mean_predicted_value = extract_calibration_data(saved_calibration_data, model_name, chosen_diag_group)
    # plot for that model and chosen diag group
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s" % (model_name, ))#,colors = plt.get_cmap(chosen_palette).colors)
    
  ax1.set_ylabel("True Probability (Fraction of Positives)")
  ax1.set_xlabel("Mean Predicted Probability")
  ax1.set_ylim([-0.05, 1.05])
  ax1.legend(loc="upper left")
  ax1.get_legend().set_visible(True)
  ax1.set_title(f'Calibration plots  (reliability curve) for Diagnosis Group {chosen_diag_group}')
  
  display(ax1)

# COMMAND ----------


from scipy.sparse import lil_matrix
import numpy as np

BYTES_TO_MB_DIV = 0.000001
def print_memory_usage_of_data_frame(df):
  """
  This function shows dataframe memory usage
  """
    mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 
    print("Memory usage is " + str(mem) + " MB")
    


def data_frame_to_scipy_sparse_matrix(df):
    """
    Converts a sparse pandas data frame to sparse scipy csr_matrix.
    :param df: pandas data frame
    :return: csr_matrix
    """
    arr = lil_matrix(df.shape, dtype=np.float32)
    for i, col in enumerate(df.columns):
        ix = df[col] != 0
        arr[np.where(ix), i] = 1

    return arr.tocsr()

def get_csr_memory_usage(matrix):
  """
  This function shows csr matrix memory usage
  """
    mem = (X_csr.data.nbytes + X_csr.indptr.nbytes + X_csr.indices.nbytes) * BYTES_TO_MB_DIV
    print("Memory usage is " + str(mem) + " MB")

    

# COMMAND ----------

from pyspark.sql.functions import concat, col, lit
def get_results_in_target_groups(dboutput:str, table:str, prefix:str, target, interested_models=False):
  """
  Get the table for either elixhauser or charlson results
  """
  
  if 'calibration' in table:
    all_model_data = spark.sql(f"""select * 
                FROM 
                (
                SELECT *,
              dense_rank() OVER (PARTITION BY model,Diag_Group ORDER BY PR_AUC_SCORE DESC) AS rank
          FROM {dboutput}.{table}
      ) vo WHERE rank = 1 and Diag_Group in {target}
       ORDER BY Diag_Group
                """)
    if interested_models != False:
      all_model_data = all_model_data.where(col('model').isin(interested_models)==True)
     
    all_model_data = all_model_data.withColumn('model' ,concat(lit(prefix), col("model")))
  else:
    all_model_data = spark.sql(f"""select * 
                FROM 
                (
                SELECT *,
              dense_rank() OVER (PARTITION BY Model_Name,Diag_Group ORDER BY PR_AUC_SCORE DESC) AS rank
          FROM {dboutput}.{table}
      ) vo WHERE rank = 1 and Diag_Group in {target}
       ORDER BY Diag_Group
                """).dropna(subset=["Date_Run"]).dropDuplicates(['Model_Name','Diag_Group'])
    all_model_data = all_model_data.where(col('Model_Name').isin(interested_models)==True)
     
    all_model_data = all_model_data.withColumn('Model_Name' ,concat(lit(prefix), col("Model_Name")))
    
  return all_model_data#, yrs_model_data


# COMMAND ----------

from pyspark.sql.functions import regexp_replace
d
ef rename_models(model_data, calibration=False):
  """
  Renames the mode;_name to be easier to read on a graph
  """
  if calibration == True:
    output = (model_data.withColumn('model', regexp_replace('model', 'Calib_XGBoost_default', 'XGB')) # rename XGboost to be viisble
          .withColumn('model', regexp_replace('model', 'Calib_LogisticRegression_l1', 'Lasso Regression')) # rename Lasso regression L1 to be visible
          .withColumn('model', regexp_replace('model', 'Calib_LogisticRegression_sag', 'Ridge Regression')) # rename Logisitic Regression L2 to be visible
          .withColumn('model', regexp_replace('model', 'Charl', 'CG')) # rename Charlson Group to be visible)
          .withColumn('model', regexp_replace('model', 'Calib_Dummy', 'DummyClassifier'))
    )
    
  else:  
    output = (model_data.withColumn('Model_Name', regexp_replace('Model_Name', 'Calib_XGBoost_default', 'XGB')) # rename XGboost to be viisble
          .withColumn('Model_Name', regexp_replace('Model_Name', 'Calib_LogisticRegression_l1', 'Lasso Regression')) # rename Lasso regression L1 to be visible
          .withColumn('Model_Name', regexp_replace('Model_Name', 'Calib_LogisticRegression_sag', 'Ridge Regression')) # rename Logisitic Regression L2 to be visible
          .withColumn('Model_Name', regexp_replace('Model_Name', 'Charl', 'CG')) # rename Charlson Group to be visible)
          .withColumn('Model_Name', regexp_replace('Model_Name', 'Calib_Dummy', 'DummyClassifier'))
    )
  return output
  
