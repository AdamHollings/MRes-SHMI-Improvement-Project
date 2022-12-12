# Databricks notebook source
# MAGIC %md
# MAGIC # Recreating SHMI
# MAGIC As part of the SHMI project for my masters project
# MAGIC 
# MAGIC Use SUSPELLID and EPIORDER in order to keep the first episode.
# MAGIC 
# MAGIC 14 09 2021 - Implemented age grouping and confirmed it works fine.
# MAGIC 
# MAGIC 23 03 2022 - added all diagnosis group mapping rather than just 42. Confirmed that they are all being mapped to the SHMI analysis data via DIAG_4_01

# COMMAND ----------

#import statements
import pandas as pd
import numpy as np

# COMMAND ----------

# DBTITLE 1,Used as reference by the conditions - diagnosis group is handled by the mapping file and joined on in the cell below/
specialist_trusts = ['RBS' ,'RQ3' ,'RP4' ,'RBQ','REP','RP6','RPC','RAN','RGM','RCU','RBV','REN','RL1','RPY','RRJ','RET']
mental_health_community_trusts = ['RVN','RRP','RWX','RXT','RYW','RY2','RT1','RYV','RV3','RYX','RXA','RJ8','RYG','RX4','RY8','RXM','RWV','RDY','RYK','RWK',
'R1L','RTQ','RXV','R1A','RY4','RWR','RY9','RV9','RXY','RYY','RW5','RGD','RY6','RT5','RY5','RP7','RW4','RRE','RMY','RY3',
 'RAT','RLY','RTV','RP1','RHA','RNU','RPG','RT2','RXE','R1D','R1C','RV5','RQY','RXG','RW1','RXX','RDR','RX2','RNK','RX3','RKL','RY7']


# COMMAND ----------

# MAGIC %md 
# MAGIC # Import HES data and define the Tier 4 codes
# MAGIC 
# MAGIC The cell below not only gathers up the data needed for the shmi analysis conditions, it also joins on the SHMI diagnosis group mapping to get the diagnosis group for each patient.

# COMMAND ----------

# DBTITLE 1,Get data from HES and apply first condition - EPISTAT = 3 and second condition ADMIDATe > 1935-01-01
Database = 'hes'
year = '2021'
OTR_Table = Database+'.'+'hes_apc_otr_' + year
APC_Table = Database+'.'+'hes_apc_' + year

RPEndDate = '2020-04-30'
YEAR_INDEX_1_lower_bound = '2019-04-01'
YEAR_INDEX_2_lower_bound = '2018-04-01'
RPStartDate = '2017-04-01'


def hes_data_grab(year):
  Database = 'hes'
  OTR_Table = Database+'.'+'hes_apc_otr_' + year
  APC_Table = Database+'.'+'hes_apc_' + year
  HES_year = spark.sql(f"""
  SELECT	a.FYEAR,
          a.PARTYEAR,
          a.EPIKEY, 
          PSEUDO_HESID,
          SUSSPELLID,
          m.BIRWEIT_1, 
          --HESID, 
          DIAG_4_01 as DIAG_1,
          DIAG_4_02 as DIAG_2,
          DIAG_4_03 as DIAG_3,
          DIAG_4_04 as DIAG_4,
          DIAG_4_05 as DIAG_5,
          DIAG_4_06 as DIAG_6,
          DIAG_4_07 as DIAG_7,
          DIAG_4_08 as DIAG_8,
          DIAG_4_09 as DIAG_9,
          DIAG_4_10 as DIAG_10,
          DIAG_4_11 as DIAG_11,
          DIAG_4_12 as DIAG_12,
          DIAG_4_13 as DIAG_13,
          DIAG_4_14 as DIAG_14,
          DIAG_4_15 as DIAG_15,
          DIAG_4_16 as DIAG_16,
          DIAG_4_17 as DIAG_17,
          DIAG_4_18 as DIAG_18,
          DIAG_4_19 as DIAG_19,
          DIAG_4_20 as DIAG_20,
          DIAG_4_CONCAT,
          OPERTN_4_CONCAT,
          CLASSPAT,
          PROCODET,
          PROCODE,
          SITETRET,
          PROVSPNOPS,
          ADMIMETH as P_SPELL_ADMIMETH, 
          DISMETH, 
          ADMISORC, 
          DISDEST, 				 
          MONTH(ADMIDATE) as ADMISSION_MONTH,
          EPISTART,
          EPIEND,
          DISDATE as P_SPELL_DISDATE,  
          EPIORDER,
          EPISTAT,
          --STARTAGE,-- Don't need now that I confirmed age grouping below works fine
          LSOA11,
          IMD04RK as IMD_RK,
          IMD04,
          IMD04C,
          IMD04ED,
          IMD04EM,
          IMD04HD,
          IMD04HS,
          IMD04I,
          IMD04IA,
          IMD04IC,
          IMD04LE,
          IMD04_DECILE,
          SEX,
          ADMIDATE as P_SPELL_ADMIDATE,
          ETHNOS,
          map.SHMI_DIAGNOSIS_GROUP,
        
        -- Age bands
        CASE 
        when STARTAGE >= 7001 and STARTAGE <= 7007 then '1'
        when STARTAGE >= 1 and STARTAGE <= 4 then '2'
        WHEN STARTAGE >= 5 and STARTAGE <= 9 THEN '3'
        WHEN STARTAGE >= 10 and STARTAGE <= 14 THEN '4'
        WHEN STARTAGE >= 15 and STARTAGE <= 19 THEN '5'
        WHEN STARTAGE >= 20 and STARTAGE <= 24 THEN '6'
        WHEN STARTAGE >= 25 and STARTAGE <= 29 THEN '7'
        WHEN STARTAGE >= 30 and STARTAGE <= 34 THEN '8'
        WHEN STARTAGE >= 35 and STARTAGE <= 39 THEN '9'
        WHEN STARTAGE >= 40 and STARTAGE <= 44 THEN '10'
        WHEN STARTAGE >= 45 and STARTAGE <= 49 THEN '11'
        WHEN STARTAGE >= 50 and STARTAGE <= 54 THEN '12'
        WHEN STARTAGE >= 55 and STARTAGE <= 59 THEN '13'
        WHEN STARTAGE >= 60 and STARTAGE <= 64 THEN '14'
        WHEN STARTAGE >= 65 and STARTAGE <= 69 THEN '15'
        WHEN STARTAGE >= 70 and STARTAGE <= 74 THEN '16'
        WHEN STARTAGE >= 75 and STARTAGE <= 79 THEN '17'
        WHEN STARTAGE >= 80 and STARTAGE <= 84 THEN '18'
        WHEN STARTAGE >= 85 and STARTAGE <= 89 THEN '19'
        WHEN STARTAGE >= 90 and STARTAGE <= 120 THEN '20'
        ELSE '21'
        END as AGE_GROUP,
        
        -- SEX column generation
        CASE
        when SEX in (0,9) then 3
        when SEX = 2 then 2
        when SEX = 1 then 1
        END as GENDER,
        
        -- ADMIMETH column banding
        CASE
        when ADMIMETH in (11,12,13) then 1
        when ADMIMETH = 99 then 2
        when ADMIMETH in (21,22,23,24,25,'2A','2B','2C','2D',28,31,32,81,82,83,84,89,98) then 3
        END as ADMIMETH         

  FROM	{APC_Table} a 
          INNER JOIN {Database+'.'+'hes_apc_otr_' + year} o ON a.EPIKEY = o.EPIKEY
          LEFT JOIN {Database+'.'+'hes_apc_mat_' + year} m ON a.EPIKEY = m.EPIKEY
          -- INNER JOIN FLAT_HES_S.dbo.HES_APC_1213 s ON a.EPIKEY = s.EPIKEY
          LEFT JOIN hes_and_hes_ons_ns_ml_collab.shmi_diag_groups map ON a.DIAG_4_01 = map.ICD10_CODE
  WHERE	
          EPISTAT = 3
          AND ADMIDATE > '1935-01-01'
          AND PROCODET is not NULL
          AND PROCODET NOT in {tuple(specialist_trusts + mental_health_community_trusts)} -- remove the specilialist trusts
          AND DIAG_4_CONCAT NOT LIKE '%U071%' -- covid exclusion condition
          --AND DISMETH not '5'
          --AND CLASSPAT not in ('2', '3', '4')
          
          
  """)
  HES_year = HES_year.where((HES_year['P_SPELL_DISDATE'] >= RPStartDate) & (HES_year['P_SPELL_DISDATE'] <= RPEndDate)) # Disdate in period condition?
  HES_year = HES_year.fillna({'P_SPELL_ADMIMETH': 99, 'SEX': '9'})
  return HES_year

HES_2021_pdf = hes_data_grab('2021').toPandas()

# COMMAND ----------

# DBTITLE 1,Grab multiple years of data with multiple spells per person
year_list = ['2021', '1920','1819']
HES_2021 = hes_data_grab('2021') # for the extra month records for people who died within 30 days
HES_1920 = hes_data_grab('1920') # records up to march 2020
HES_1819 = hes_data_grab('1819')
HES_1718 = hes_data_grab('1718') # for records back down to 1st April 2017


from functools import reduce
from pyspark.sql import DataFrame

HES_years = [HES_2021,HES_1920,HES_1819, HES_1718]
SHMI_HES_cohort = reduce(DataFrame.unionAll, HES_years)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Link to HES ONS
# MAGIC Also skipped the CAUSE OF DEATH covid flag as thats in the HES ONS

# COMMAND ----------

HES_ONS = spark.table("ons.vw_hes_ons").drop('HESID', 'DATA_SOURCE', 'DATE_ADDED', 'DOR', 'SUBSEQUENT_ACTIVITY', 'COMMUNAL_ESTABLISHMENT', 'NHS_INDICATOR', 'MATCH_RANK', 'RESSTHA', 'RESPCT')

# COMMAND ----------

SHMI_HES_ONS_cohort = SHMI_HES_cohort.join(HES_ONS[['DOD', 'PSEUDO_HESID']], on='PSEUDO_HESID', how='left')

# COMMAND ----------

# DBTITLE 1,Get the year indexes
#spark.catalog.dropTempView("SHMI_HES_cohort_temp")
SHMI_HES_ONS_cohort.createTempView('SHMI_HES_ONS_cohort_temp')
year_index_query = f"""
 select *, 
 CASE 
 WHEN P_SPELL_DISDATE < '{RPEndDate}' and P_SPELL_DISDATE > '{YEAR_INDEX_1_lower_bound}' THEN 1
 WHEN P_SPELL_DISDATE < '{YEAR_INDEX_1_lower_bound}' and P_SPELL_DISDATE > '{YEAR_INDEX_2_lower_bound}' THEN 2
 WHEN P_SPELL_DISDATE < '{YEAR_INDEX_2_lower_bound}' and P_SPELL_DISDATE > '{RPStartDate}' THEN 3
 END as YEAR_INDEX,
 
  ----Birthweigth grouping
 CASE
 when AGE_GROUP = 1 AND BIRWEIT_1 < 1000 then '1'
 when AGE_GROUP = 1 AND BIRWEIT_1 >= 1000 AND BIRWEIT_1 < 2500 then '2'
 when AGE_GROUP = 1 AND BIRWEIT_1 >= 2500 AND BIRWEIT_1 < 4500 then '3'
 when AGE_GROUP = 1 AND BIRWEIT_1 >= 4500 then '4'
 when AGE_GROUP = 1 AND BIRWEIT_1 is NULL then '5'
 ELSE '6'
 END as BIRWEIT_GROUP,
 
 -- DIED column creation
 CASE
 when (DATEDIFF(DOD, P_SPELL_DISDATE) < 31) AND (P_SPELL_ADMIDATE <= DOD) then '1'
 ELSE '0'
 END as DIED 
 
 from SHMI_HES_ONS_cohort_temp"""
SHMI_HES_ONS_cohort = spark.sql(year_index_query)

# COMMAND ----------

# MAGIC %md
# MAGIC # Export to table
# MAGIC 
# MAGIC ## Below the export is all checking script I used to make sure the data was sensible

# COMMAND ----------

export_database = 'hes_and_hes_ons_ns_ml_collab'
SHMI_HES_ONS_cohort.write.mode("overwrite").saveAsTable(f"{export_database}.shmi_analysis_bodge")
spark.sql(f"""
ALTER TABLE {export_database}.shmi_analysis_bodge OWNER TO {export_database}""")

# COMMAND ----------

SHMI_HES_ONS_cohort.count()
