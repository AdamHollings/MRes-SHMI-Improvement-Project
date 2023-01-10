# Using alternative modelling methods and more granular diagnosis encoding for the Summary Hospital-level Mortality Indicator (SHMI) 
An MRes final project for Adam Hollings on improving the Summary Hospital Mortality Indicator (SHMI) funded through my employer, NHS Digital (soon NHS England) at the University of Leeds. 

By Adam Hollings (adam.hollings1@nhs.net)

## Abstract
### Objectives:
SHMI does not score well when modelling some diagnosis groups despite using ROC AUC score which is forgiving when class imbalance is high. This study aimed to explore the effect on scores of replacing the Charlson index score with up to 260 one hot encoded diagnosis features as well as testing lasso regression and XGboost instead of the current SHMI ridge regression models. It also used PR AUC score instead of ROC AUC score.
### Design:
We examined three groups PR AUC and found the more granular approach had better performance: Septicemia and Shock (SHMI 0.3, CCS groups 0.6, Elixhauser groups 0.4, Charlson groups 0.4), HIV (SHMI 0.1, CCS groups 0.4, Elixhauser groups 0.1, Charlson groups 0.1), Alcohol Related Liver Disease  (SHMI 0.2, CCS groups 0.6, Elixhauser groups 0.4, Charlson groups 0.3).
### Setting:
1st April 2017 and 30th April 2020 England, United Kingdom.
### Participants:
709 sites across 122 NHS trusts across England were used for this study 
### Results:
PR AUC scores were lower overall compared to ROC AUC scores. The results showed the greater the granularity of diagnosis encoding the higher the PR AUC score, with the most granular having approximately double to eight times the score of the SHMI model for the same group for the groups studied.
### Conclusions:
SHMI could be significantly improved by considering more granular diagnosis encoding methods. PR AUC should be used as a scoring method instead of ROC AUC.

## Strengths and Limitations of this study
- Used PR AUC unlike the current SHMI implementation and so better represents how well both deaths and survival are predicted by the models
- Large sample size
- This study only covers 3 diagnosis groupings, although the supplementary information covers 139 other diagnosis groupings.
- Does not include ethnicity or deprivation as features
- Diagnosis representation categories were given equal weighting at feature creation stage


