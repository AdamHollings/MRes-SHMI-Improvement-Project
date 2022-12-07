# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC This runs through all groups for the chosen model, and saves the model fit stats and calibration numbers so they can be plotted later

# COMMAND ----------

# DBTITLE 1,Go through all groups
from sklearn.calibration import calibration_curve
from datetime import date

fit_model_storage = []
coefficient_storage = []
collected_models = []
collected_models_means = []

for number in ok_to_run_list_diag_groups:
  group = number
  print("Group:", group)
  # create the holdout, and the train data 
  X, Y, X_train, X_test, Y_train, Y_test = create_holdout(SHMI_data_all_groups, group)
  
  # Do the 5 fold cross validation  
  grid_scores, Y_pred, Y_pred_proba, fit_model = run_model_score(**list_of_models[model_name])
  grid_scores['Diag_Group'] = group
  
  #collect the grid_scores
  collected_models.append(grid_scores)
  # save the fit model to show the stats later
  fit_model_storage = {"Group" : group, "fit_model" : fit_model, "Y_test" : Y_test, "Y_pred_proba" : Y_pred_proba, "Y_pred": Y_pred}
  
  #show the grid scores
  print(pd.DataFrame(grid_scores))#.drop(columns=["mean_fit_time", 	"std_fit_time",	"mean_score_time",	"std_score_time"])
  grid_scores['Diag_Group'] = group
  grid_scores['Model_Name'] = model_name
  grid_scores['Date_Run'] = date.today() # from datetime import date
  spark.createDataFrame(grid_scores).write.format('delta').mode("append").saveAsTable(f"{dboutput}.{table}_training")
  spark.sql(f""" ALTER TABLE {dboutput}.{table}_training OWNER TO {dboutput}""")
  
  # show the output and print to the server
  score_output = pandas_score_output(model_name, Y_test = Y_test, Y_pred_proba = Y_pred_proba)
  score_output['Diag_Group'] = group
  score_output['Date_Run'] = date.today() # from datetime import date
  spark.createDataFrame(score_output).write.format('delta').mode("append").saveAsTable(f"{dboutput}.{table}")
  spark.sql(f""" ALTER TABLE {dboutput}.{table} OWNER TO {dboutput}""")
  print(score_output)
  
#   if ("XGBoost" not in model_name) and ( "Dec_Tree" not in model_name):
#     #save coefficients to a dataframe
#     coefficient_storage.append(show_coefficients(fit_model=fit_model.named_steps['model'], X=X).T)
  
  #save calibration surve stuff 
  
  fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, Y_pred_proba, n_bins=10)
  
  columns = ["model","diag_group","fraction_of_positives", "mean_predicted_value", "PR_AUC_SCORE"]
  data = [(model_name, group, str(fraction_of_positives), str(mean_predicted_value), float(score_output.iloc[0]['PR_AUC_SCORE']))]

  saved_calibration_data = spark.createDataFrame(data).toDF(*columns)
  saved_calibration_data.write.format('delta').mode("append").saveAsTable(f"{dboutput}.{calibration_table}")
  spark.sql(f""" ALTER TABLE {dboutput}.{calibration_table} OWNER TO {dboutput}""")
  
  
