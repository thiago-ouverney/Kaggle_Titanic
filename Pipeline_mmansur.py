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
    df.drop(columns=['Name' ,'Ticket' ,'Cabin' ,'Embarked' ,'Sex' ,'Pclass'], inplace=True)
    return df

# dealing with missing numbers
def dealing_null_values(df):
    df = df['Age'].fillna(-1)
    return df