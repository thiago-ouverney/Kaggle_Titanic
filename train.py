import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

#Model_Selection
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv("Dados/train.csv",index_col=0)
x = df.drop("Survived",axis=1).copy()
y = df.Survived

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


get_drop_columns_with_null_valuse = FunctionTransformer(drop_columns_with_null_valuse,validate=False)
get_drop_categorical_features = FunctionTransformer(drop_categorical_features,validate=False)


x_train, x_val, y_train, y_val = train_test_split(x,y,
                                                    test_size = 0.3,
                                                    random_state = 0)


pipe1 = Pipeline(memory=None,
                 steps = [
                     ("Feature_Selection",get_drop_categorical_features),
                     ("Null_Validate",get_drop_columns_with_null_valuse),
                     ("RandomForest", RandomForestClassifier() )
                 ],
                verbose=False
                )


pipe1.fit(x_train,y_train)
test_score = pipe1.score(x_val,y_val)
with open("metrics.txt", 'w') as outfile:
    outfile.write(f"Test Score: {test_score}")




