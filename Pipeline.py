import pandas as pd, numpy as np
# PIPELINE
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.impute import SimpleImputer
import warnings
# MODELS
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
#VARIABLES
warnings.filterwarnings("ignore")
seed = 0

################# PIPELINE DADOS #################

def name_information(df):
    df["Name_aux"] = df["Name"]
    df["Name_aux"] = df["Name_aux"].str.replace(",|\.|\(|\)|\"|Mrs|Mr|Miss", "", regex=True)
    df["Name_aux"] = df["Name_aux"].str.replace("\W[A-Z]\W", "", regex=True)
    df["Name_aux"] = df["Name_aux"].str.replace("\W[A-Z]$", "", regex=True)
    df["Name_aux"] = df["Name_aux"].str.replace("  ", " ", regex=True)
    df["Name_List"] = df["Name_aux"].apply(lambda x: x.split(" "))
    df["Last_Name"] = df["Name_List"].apply(lambda x: x[-1])
    df.drop(["Name","Name_List"], axis=1, inplace=True)
    return df

def cabin_information(df):
    df['Cabin'].fillna("S",inplace=True)
    df['Category_Cabin'] = df['Cabin'].apply(lambda x: x[0])
    df['Size_Cabin'] = df['Cabin'].apply(lambda x: len(x.split(" ")))
    df.drop("Cabin",axis=1,inplace=True)
    return df

def ticket_information(df):
    #df.drop("Ticket",axis=1,inplace=True)
    return df



class FeatureEngPipe(BaseEstimator):

    def __init__(self,name=True,cabin=True,ticket=True):
        self.name = name
        self.cabin = cabin
        self.ticket = ticket
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, x_dataset):
        if self.name:
            x_dataset = name_information(x_dataset)

        if self.cabin:
            x_dataset = cabin_information(x_dataset)

        if self.ticket:
            x_dataset = ticket_information(x_dataset)
        return x_dataset

#pipeline for columns transformations on categorical features
cat_preprocessing = make_pipeline( SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
                                    OneHotEncoder(handle_unknown='ignore') ) #Só vai fazer no test data o q já fez no train ou em inf no test)

num_preprocessing = make_pipeline( SimpleImputer(missing_values=np.nan, strategy='median'))
pipe_preprosseging = ColumnTransformer( [("numeric_transf", num_preprocessing, make_column_selector(dtype_exclude=object)),    # NOME-PROCESSSO  $$$$$ TRANFORMACAO A SER APLCIADA $$$$$ COLUNAS QUE VAO SOFRER A TRANF.
                                        ("categorical_transf", cat_preprocessing, make_column_selector(dtype_include=object))])


pipe_RF = Pipeline(memory=None,
                      steps = [
                          ("FeatureEng",FeatureEngPipe()),
                          ("Fixing_Missing_Values_One_Hot_Enconder", pipe_preprosseging),
                          ("RandomForest", RandomForestClassifier(random_state=seed) )
                      ]
                      )

pipe_GB = Pipeline(memory=None,
                      steps = [
                          ("FeatureEng",FeatureEngPipe()),
                          ("Fixing_Missing_Values_One_Hot_Enconder", pipe_preprosseging),
                          ("Gradient_Boosting", GradientBoostingClassifier(random_state=seed) )
                      ]
                      )

pipe_XGBoost = Pipeline(memory=None,
                      steps = [
                          ("FeatureEng",FeatureEngPipe()),
                          ("Fixing_Missing_Values_One_Hot_Enconder", pipe_preprosseging),
                          ("XGBoost", XGBClassifier(random_state=seed,eval_metric='error'))
                      ]
                      )