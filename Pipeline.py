import pandas as pd, numpy as np
import warnings

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

#VARIABLES
warnings.filterwarnings("ignore")
seed = 42

################# PIPELINE DADOS #################

def preprosseging_dtypes(df2):
    df = df2.copy()
    df.Pclass = df.Pclass.astype("object")
    return df

def name_information(df2):
    df = df2.copy()
    df["Name_aux"] = df["Name"]
    df["Name_aux"] = df["Name_aux"].str.replace(",|\.|\(|\)|\"|Mrs|Mr|Miss", "", regex=True)
    df["Name_aux"] = df["Name_aux"].str.replace("\W[A-Z]\W", "", regex=True)
    df["Name_aux"] = df["Name_aux"].str.replace("\W[A-Z]$", "", regex=True)
    df["Name_aux"] = df["Name_aux"].str.replace("  ", " ", regex=True)
    df["Name_List"] = df["Name_aux"].apply(lambda x: x.split(" "))
    df["Last_Name"] = df["Name_List"].apply(lambda x: x[-1])
    df.drop(["Name","Name_List"], axis=1, inplace=True)
    return df

def cabin_information(df2):
    df = df2.copy()
    df['Cabin'].fillna("S",inplace=True)
    df['Category_Cabin'] = df['Cabin'].apply(lambda x: x[0])
    df['Size_Cabin'] = df['Cabin'].apply(lambda x: len(x.split(" ")))
    df.drop("Cabin",axis=1,inplace=True)
    return df
def ticket_information(df2):
    df = df2.copy()
    #df.drop("Ticket",axis=1,inplace=True)
    return df



class FeatureEngPipe(BaseEstimator):

    def __init__(self,name=True,cabin=True,ticket=True, preprop=True):
        self.name = name
        self.cabin = cabin
        self.ticket = ticket
        self.preprop = preprop
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, x_dataset):
        if self.preprop:
            x_dataset= preprosseging_dtypes(x_dataset)

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


#### MODELS
clf1 = LogisticRegression(random_state=seed)
clf2 = RandomForestClassifier(random_state=seed)
clf3 = GradientBoostingClassifier(random_state=seed)
clf4 = XGBClassifier(random_state=seed,eval_metric='error')



#### PIPELINES
pipe_RF = Pipeline(memory=None,
                      steps = [
                          ("FeatureEng",FeatureEngPipe()),
                          ("Fixing_Missing_Values_One_Hot_Enconder", pipe_preprosseging),
                          ("RandomForest", clf2 )
                      ]
                      )

pipe_GB = Pipeline(memory=None,
                      steps = [
                          ("FeatureEng",FeatureEngPipe()),
                          ("Fixing_Missing_Values_One_Hot_Enconder", pipe_preprosseging),
                          ("Gradient_Boosting", clf3 )
                      ]
                      )

pipe_XGBoost = Pipeline(memory=None,
                      steps = [
                          ("FeatureEng",FeatureEngPipe()),
                          ("Fixing_Missing_Values_One_Hot_Enconder", pipe_preprosseging),
                          ("XGBoost", clf4)
                      ]
                      )


pipe_VotingClassifier_Soft = Pipeline(memory=None,
                      steps = [
                          ("FeatureEng",FeatureEngPipe()),
                          ("Fixing_Missing_Values_One_Hot_Enconder", pipe_preprosseging),
                          ("Voting", VotingClassifier(voting='soft',weights= [1,2,3] ,estimators=[('lr', clf1), ('rf', clf2), ('gbc', clf3)]))
                      ]
                      )


pipe_VotingClassifier_Hard = Pipeline(memory=None,
                      steps = [
                          ("FeatureEng",FeatureEngPipe()),
                          ("Fixing_Missing_Values_One_Hot_Enconder", pipe_preprosseging),
                          ("Voting", VotingClassifier(voting='hard',estimators=[('lr', clf1), ('rf', clf2), ('gbc', clf3)]))
                      ]
                      )