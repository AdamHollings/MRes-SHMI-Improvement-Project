# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Universal Pre-process, for all Diagnosis groups - Charlson Groups - No Ethnicity
# MAGIC 
# MAGIC This notebook universally pre-processes the data for all diagnosis groups, then saves it as a table. Ethnicity removed as did not get sign off from NHS D
# MAGIC 
# MAGIC Steps taken are:
# MAGIC 
# MAGIC 1. Column selection and drop NA for all NULLS
# MAGIC 2. Group into Charlson Groups - Evidence that charlson index is inadequate
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
# MAGIC # Group into Charlson Groups

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import DataFrame as df

#SHMI_analysis_bodge.withColumn('Peptic ucler', when(F.col('DIAG_4_CONCAT').))
charlson_group_list_dict = dict( #This list was created from the SHMI specification
    acute_myocardial_infarction = ['I21', 'I22', 'I23', 'I252', 'I258']
  , cerebral_vascular_accident = ['G450', 'G451', 'G452', 'G454', 'G458', 'G459', 'G46', 'I60','I61','I62','I63','I65','I66','I67','I68','I69']
  , congestive_heart_failure = ['I50']
  , connective_tissue_disorder = ['M05', 'M060', 'M063', 'M069', 'M32', 'M332', 'M34', 'M353']
  , dementia = ['F00', 'F01', 'F02', 'F03', 'F051']
  , diabetes = ['E101', 'E105', 'E106', 'E108', 'E109', 'E111', 'E115', 'E116', 'E118', 'E119', 'E131', 'E135', 'E136', 'E138', 'E139', 'E141', 'E145', 'E146', 'E148', 'E149']
  , liver_disease = ['K702', 'K703', 'K717', 'K73', 'K74'] 
  , peptic_ulcer = ['K25' , 'K26', 'K27', 'K28']
  , peripheral_vascular_disease = ['I71', 'I739', 'I790', 'R02', 'Z958', 'Z959']
  , pulmonary_disease = ['J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47','J60','J61','J62','J63','J64','J65','J66','J67','J68','J69','J70']
  , cancer = ['C00' ,'C01' ,'C02' ,'C03' ,'C04' ,'C05' ,'C06' ,'C07' ,'C08' ,'C09' ,'C10' ,'C11' ,'C12' ,'C13' ,'C14' ,'C15' ,'C16' ,'C17' ,'C18' 
              ,'C19' ,'C20' ,'C21' ,'C22' ,'C23' ,'C24' ,'C25' ,'C26' ,'C30' ,'C31' ,'C32' ,'C33' ,'C34' ,'C37' ,'C38' ,'C39' ,'C40' ,'C41' ,'C43'
              ,'C44' ,'C4A' ,'C45' ,'C46' ,'C47' ,'C48' ,'C49' ,'C50' ,'C51' ,'C52' ,'C53' ,'C54' ,'C55' ,'C56' ,'C57' ,'C58' ,'C60' ,'C61' ,'C62' 
              ,'C63' ,'C64' ,'C65' ,'C66' ,'C67' ,'C68' ,'C69' ,'C70' ,'C71' ,'C72' ,'C73' ,'C74' ,'C75','C81','C82','C83','C84','C85','C86','C88'
              ,'C90','C91','C92','C93','C94','C95','C96', 'C97'] #maybe C97 not needed?
  , diabetes_complications = ['E102', 'E103', 'E104', 'E107', 'E112', 'E113', 'E114', 'E117', 'E132', 'E133', 'E134', 'E137', 'E142', 'E143', 'E144', 'E147']
  , paraplegia = ['G041', 'G81', 'G820', 'G821', 'G822']
  , renal_disease = ['I12', 'I13', 'N01', 'N03', 'N052', 'N053', 'N054', 'N055', 'N056', 'N072', 'N073', 'N074', 'N18', 'N19', 'N25']
  , metastatic_cancer = ['C77', 'C78', 'C79', 'C80']
  , severe_liver_disease = ['K721', 'K729', 'K766', 'K767']
  , hiv = ['B20', 'B21', 'B22', 'B23', 'B24', 'O987']
    )
charlson_group_conditions_list = list(charlson_group_list_dict)

def charlson_grouper(data: df, code_list: list, column_name: str) -> df:
    """
    This function tags records with their presence in a charlson group or not 
    (the 17 condition groupings used in the Charlson index)
    
    Args: dataset we are reading from as a spark dataframe, the list of codes 
    for that charlson group and  the name of the column that will be created 
    with that tag.
    
    Returns: a spark dataframe with the new column added
    
    """
    return data.select(
      "*", 
     F.when(col("DIAG_4_CONCAT").rlike("|".join(code_list)), "1").otherwise("0").alias(column_name)
      )

from pyspark.sql.types import IntegerType,BooleanType
#Apply the charlson grouper to tag all records with their presence in a group or not. Then cast them to boolean
for condition in charlson_group_conditions_list:
  SHMI_analysis_bodge = charlson_grouper(SHMI_analysis_bodge, charlson_group_list_dict[condition], condition)
  SHMI_analysis_bodge = SHMI_analysis_bodge.withColumn(condition, SHMI_analysis_bodge[condition].cast(BooleanType()))

unbalanced_class_df = SHMI_analysis_bodge.drop('DIAG_4_CONCAT')


# COMMAND ----------

# DBTITLE 1,Fill any nulls
SHMI_data = unbalanced_class_df.fillna(0.0, 'YEAR_INDEX')

# COMMAND ----------

# DBTITLE 1,Save to a table
export_database = 'hes_and_hes_ons_ns_ml_collab'
name = 'shmi_bodge_charlson_groups_no_imd'
SHMI_data.write.mode("overwrite").saveAsTable(f"{export_database}.{name}")
spark.sql(f"""
ALTER TABLE {export_database}.{name} OWNER TO {export_database}""")

# COMMAND ----------

# DBTITLE 1,Check the number of rows in the saved table
spark.table(f"{export_database}.{name}").select("SUSSPELLID").count()
