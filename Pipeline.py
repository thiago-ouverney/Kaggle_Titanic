import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

#Model_Selection
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt
import seaborn as sns

def drop_columns_with_null_valuse(df):
    df.dropna(axis=1,inplace=True)
    try:
        df.drop("Fare",axis=1,inplace=True)
    except:
        pass
    return df

def drop_categorical_features(df):
    df = df.select_dtypes(exclude="object")
    return df
def saving_columns(df):
    global colunas
    colunas= df.columns
    return df



get_drop_columns_with_null_valuse = FunctionTransformer(drop_columns_with_null_valuse,validate=False)
get_drop_categorical_features = FunctionTransformer(drop_categorical_features,validate=False)
get_colums_names = FunctionTransformer(saving_columns,validate=False)


