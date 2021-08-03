import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer,make_column_transformer,make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")
seed = 0

################# PIPELINE NOVO #################

def name_information(df):
    df["Name"] = df["Name"].str.replace(",|\.|\(|\)|\"|Mrs|Mr|Miss", "", regex=True)
    df["Name"] = df["Name"].str.replace("\W[A-Z]\W", "", regex=True)
    df["Name"] = df["Name"].str.replace("\W[A-Z]$", "", regex=True)
    df["Name"] = df["Name"].str.replace("  ", " ", regex=True)
    df["Name_List"] = df["Name"].apply(lambda x: x.split(" "))
    df["Last_Name"] = df["Name_List"].apply(lambda x: x[-1])
    df.drop(["Name", "Name_List"], axis=1, inplace=True)
    return df
get_name_inf = FunctionTransformer(name_information,validate=False)

def cabin_information(df):
    df['Cabin'].fillna("S",inplace=True)
    df['Category_Cabin'] = df['Cabin'].apply(lambda x: x[0])
    df['Size_Cabin'] = df['Cabin'].apply(lambda x: len(x.split(" ")))
    df.drop("Cabin",axis=1,inplace=True)
    return df
get_cabin_inf = FunctionTransformer(cabin_information,validate=False)


#pipeline for columns transformations on categorical features
cat_preprocessing = make_pipeline( SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
                                    OneHotEncoder(handle_unknown='ignore')  #Só vai fazer no test data o q já fez no train ou em inf no test
                                   )
num_preprocessing = make_pipeline( SimpleImputer(missing_values=np.nan, strategy='median'))


pipe_preprosseging = ColumnTransformer( [("numeric_transf", num_preprocessing, make_column_selector(dtype_exclude=object)),    # NOME-PROCESSSO  $$$$$ TRANFORMACAO A SER APLCIADA $$$$$ COLUNAS QUE VAO SOFRER A TRANF.
                                        ("categorical_transf", cat_preprocessing, make_column_selector(dtype_include=object))]
                                        )


pipe_RF = Pipeline(memory=None,
                      steps = [
                          ("FE_Name",get_name_inf),
                          ("FE_Cabin" , get_cabin_inf),
                          ("Fixing_Missing_Values_One_Hot_Enconder", pipe_preprosseging),
                          ("RandomForest", RandomForestClassifier(random_state=seed) )
                      ]
                      )

pipe_GB = Pipeline(memory=None,
                      steps = [
                          ("FE_Name",get_name_inf),
                          ("FE_Cabin" , get_cabin_inf),
                          ("Fixing_Missing_Values_One_Hot_Enconder", pipe_preprosseging),
                          ("Gradient_Boosting", GradientBoostingClassifier(random_state=seed) )
                      ]
                      )