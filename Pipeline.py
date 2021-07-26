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
    
# function to be applied in each row of the "Sex" column
# and change object data to categorical (1 for male, 0 for female"
def sex_to_binary(n):
    if n == 'male':
        return 1
    elif n == 'female':
        return 0


def drop_columns_with_null_valuse(df):
    df.dropna(axis=1, inplace=True)
    try:
        df.drop("Fare", axis=1, inplace=True)
    except:
        pass
    return df


def drop_categorical_features(df):
    df = df.select_dtypes(exclude="object")
    return df


def one_hot_encoder(df,columns):
    for column in columns:
        df_aux = pd.get_dummies(df[column],prefix=column)
        df = df.join(df_aux)
        df.drop(column,axis=1,inplace=True)
    return df

# dealing with missing numbers
def dealing_null_values(df):
    df['Age'] = df['Age'].fillna(-1)
    return df

# transform  all columns necessary
def transform_dtype_new(df,teste="ok"):
    df = one_hot_encoder(df, ["Sex","Pclass","Embarked"])
    df.drop(['Name' ,'Ticket' ,'Cabin'], axis=1, inplace=True)
    if teste != None:
        df["Fare"] = df["Fare"].fillna(0)

    return df

get_drop_columns_with_null_valuse = FunctionTransformer(drop_columns_with_null_valuse, validate=False)
get_drop_categorical_features = FunctionTransformer(drop_categorical_features, validate=False)

get_dealing_null_values = FunctionTransformer(dealing_null_values,validate=False)
get_transform_dtype_new = FunctionTransformer(transform_dtype_new,validate=False)


