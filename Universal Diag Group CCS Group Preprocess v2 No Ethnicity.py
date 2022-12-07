# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Universal Pre-process, for all Diagnosis groups - ccs Groups - No Ethnicity
# MAGIC 
# MAGIC This notebook universally pre-processes the data for all diagnosis groups, then saves it as a table. Ethnicity removed as did not get sign off from NHS D
# MAGIC 
# MAGIC Steps taken are:
# MAGIC 
# MAGIC 1. Column selection and drop NA for all NULLS
# MAGIC 2. Group into ccs Groups - Evidence that charlson index is inadequate
# MAGIC 
# MAGIC Then save to a table
# MAGIC 
# MAGIC # Issues:
# MAGIC 
# MAGIC All nulls are dropped. Might be loosing valuable data, also loosing a key factor

# COMMAND ----------

from pyspark.sql.functions import col
#import statements

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# COMMAND ----------

# DBTITLE 1,Parameters
oversample = False  #True = Oversample, False = default distribution, 'undersample' = undersample

# COMMAND ----------

columns_to_select =["SHMI_DIAGNOSIS_GROUP",
                    'SUSSPELLID',
                    'EPIORDER'
                  ,'AGE_GROUP'
                  ,'ADMIMETH'
                  ,'GENDER'
                  ,'BIRWEIT_GROUP'
                  ,'ADMISSION_MONTH'
                  ,'YEAR_INDEX'
                  ,'DIAG_4_CONCAT'
                  ,'DIED']

SHMI_analysis_bodge = spark.table("hes_and_hes_ons_ns_ml_collab.shmi_analysis_bodge").select(columns_to_select)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Group into ccs groupss

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import DataFrame as df


ccs_group_mapping = spark.table('hes_and_hes_ons_ns_ml_collab.shmi_diag_groups').toPandas()

ccs_group_mapping['CCS_CATEGORY'] = 'CCS_' + ccs_group_mapping['CCS_CATEGORY'].astype(str)
ccs_group_mapping_dict = dict(ccs_group_mapping.groupby('CCS_CATEGORY')['ICD10_CODE'].apply(list))

ccs_group_conditions_list = list(ccs_group_mapping_dict)

def ccs_grouper(data: df, code_list: list, column_name: str) -> df:
    """
    This function tags records with their presence in a ccs groups or not 
    (the 17 condition groupings used in the Charlson index)
    
    Args: dataset we are reading from as a spark dataframe, the list of codes 
    for that ccs groups and  the name of the column that will be created 
    with that tag.
    
    Returns: a spark dataframe with the new column added
    
    """
    return data.select(
      "*", 
     F.when(col("DIAG_4_CONCAT").rlike("|".join(code_list)), "1").otherwise("0").alias(column_name)
      )

from pyspark.sql.types import IntegerType,BooleanType
#Apply the ccs grouper to tag all records with their presence in a group or not. Then cast them to boolean
for condition in ccs_group_conditions_list:
  SHMI_analysis_bodge = ccs_grouper(SHMI_analysis_bodge, ccs_group_mapping_dict[condition], condition)
  SHMI_analysis_bodge = SHMI_analysis_bodge.withColumn(condition, SHMI_analysis_bodge[condition].cast(BooleanType()))

unbalanced_class_df = SHMI_analysis_bodge.drop('DIAG_4_CONCAT')


# COMMAND ----------

# DBTITLE 1,Fill any nulls
SHMI_data = unbalanced_class_df.fillna(0.0, 'YEAR_INDEX')

# COMMAND ----------

# DBTITLE 1,Save to a table
export_database = 'hes_and_hes_ons_ns_ml_collab'
name = 'shmi_bodge_ccs_groups_no_imd'
SHMI_data.write.mode("overwrite").saveAsTable(f"{export_database}.{name}")
spark.sql(f"""
ALTER TABLE {export_database}.{name} OWNER TO {export_database}""")

# COMMAND ----------

# DBTITLE 1,Check how many rows in the saved table
spark.table(f"{export_database}.{name}").select("SUSSPELLID").count()
