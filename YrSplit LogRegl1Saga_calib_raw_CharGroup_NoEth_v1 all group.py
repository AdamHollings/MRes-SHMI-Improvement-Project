# Databricks notebook source
# MAGIC %md
# MAGIC # Year Split Calibrated LogReg l1 saga (lasso regression) model Raw unbalanced data Charlson Groups No Ethnicity v1
# MAGIC This notebook is to test XGBoost on the data, using charlson groups and without Ethnicity
# MAGIC 
# MAGIC This is the same as the calibrated Xgboost default, except it is split by year rather than a stratifed split across the 3 years.
# MAGIC It saves the output to the table below for comparison

# COMMAND ----------

dboutput = 'hes_and_hes_ons_ns_ml_collab'
table = 'adam_shmi_improvement_model_outputs_all_groups_no_ethnicity_yrsplit'
calibration_table = "adam_shmi_all_groups_no_ethnicity_calibration_yrsplit"

model_name = "Calib_LogisticRegression_l1"

#Line to avoid avoid creating table issue
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")

# COMMAND ----------

# MAGIC %run "/Users/adam.hollings1@nhs.net/SHMI Project/x_y_years_holdout"

# COMMAND ----------

# DBTITLE 1,Import the data

SHMI_data_all_groups = spark.table("hes_and_hes_ons_ns_ml_collab.shmi_bodge_charlson_groups_no_imd")

# COMMAND ----------

# MAGIC %run "/Users/adam.hollings1@nhs.net/SHMI Project/run_model_score function SHMI"

# COMMAND ----------

# DBTITLE 1,List of diag groups that won't crash the cluster
too_big_groups = []
# restart from the number in the title
diag_group_list = [2,5,93]# list(range(1,143))#

ok_to_run_list_diag_groups = list(set(diag_group_list) - set(too_big_groups))
ok_to_run_list_diag_groups.sort()

# COMMAND ----------

# MAGIC %run "/Users/adam.hollings1@nhs.net/SHMI Project/All Diag Groups (CCS)/All_Groups (CCS)/all_groups_run"
