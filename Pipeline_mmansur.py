import pandas as pd
from sklearn.preprocessing import FunctionTransformer

# function to be applied in each row of the "Sex" column
# and change object data to categorical (1 for male, 0 for female"
def sex_to_binary(n):
    if n == 'male':
        return 1
    elif n == 'female':
        return 0

#transform Pclass int stype to categorical through get_dummies
def Pclass_onehot(df):
    df_Pclass_enc = pd.get_dummies(df['Pclass'])
    return df_Pclass_enc

# transform  all columns necessary
def transform_dtype(df):
    df['Gender_binary'] = df['Sex'].map(sex_to_binary)
    Pclass_dummies = Pclass_onehot(df)
    df = df.join(Pclass_dummies)
    df.drop(['Name' ,'Ticket' ,'Cabin' ,'Embarked' ,'Sex' ,'Pclass'], axis=1, inplace=True)
    return df

# dealing with missing numbers
def dealing_null_values(df):
    df['Age'] = df['Age'].fillna(-1)
    return df

def saving_columns(df):
    global colunas
    colunas= df.columns
    return df

get_dealing_null_values = FunctionTransformer(dealing_null_values,validate=False)
get_transform_dtype = FunctionTransformer(transform_dtype,validate=False)
get_colums_names = FunctionTransformer(saving_columns,validate=False)

#-----------------------------------------

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


def saving_columns(df):
    global colunas
    colunas = df.columns
    return df


get_drop_columns_with_null_valuse = FunctionTransformer(drop_columns_with_null_valuse, validate=False)
get_drop_categorical_features = FunctionTransformer(drop_categorical_features, validate=False)


