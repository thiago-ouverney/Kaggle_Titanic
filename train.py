import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from Pipeline import *



df = pd.read_csv("Dados/train.csv",index_col=0)
x = df.drop("Survived",axis=1).copy()
y = df.Survived
seed = 0


x_train, x_val, y_train, y_val = train_test_split(x,y,
                                                    test_size = 0.3,
                                                    random_state = seed)



pipe_RF = Pipeline(memory=None,
                 steps = [
                     ("Feature_Selection:get_transform_dtype",get_transform_dtype),
                     ("Null_Validate:get_dealing_null_values",get_dealing_null_values),
                     ("Final_Columns:get_colums_names",get_colums_names),
                     ("RandomForest", RandomForestClassifier(random_state=seed) )
                 ],
                verbose=False
                )

RF_params = {
    'RandomForest__criterion': ["gini", "entropy"],
    'RandomForest__bootstrap': [True,False],
    'RandomForest__min_samples_leaf': [1,2, 5, 10],
    'RandomForest__max_depth': [5, 8, 15, 25, 30],
    'RandomForest__n_estimators': [100, 200, 500, 1000]
}

RF_best_params = {'RandomForest__bootstrap': [False],
 'RandomForest__criterion': ['gini'],
 'RandomForest__max_depth': [25],
 'RandomForest__min_samples_leaf': [10],
 'RandomForest__n_estimators': [500]}

pipe_LR = Pipeline(memory=None,
                 steps = [
                     ("Feature_Selection:get_transform_dtype",get_transform_dtype),
                     ("Null_Validate:get_dealing_null_values",get_dealing_null_values),
                     ("Final_Columns:get_colums_names",get_colums_names),
                     ("ModelLR", LogisticRegression(random_state=seed) )
                 ],
                verbose=False
                )



#Salvando Scores
modelos_testados = {"Modelos":["RandomForestClassifier"],
                    "Pipeline":[pipe_RF],
                    "Score":[],
                    "Steps":[],
                    "Params":[]
                    }


with open("metrics.txt", 'w') as outfile:
    pipe = modelos_testados["Pipeline"][0]
    clf = GridSearchCV(pipe, RF_best_params,verbose=3,cv=5)
    clf.fit(x_train, y_train)


    test_score = clf.score(x_val, y_val)
    steps = clf.estimator.named_steps.keys()
    params = clf.best_params_
    nome_modelo = "RandomForestClassifier"
    outfile.write(f"{nome_modelo}- Test Score: {test_score} - Steps: {steps}")

    modelos_testados["Score"].append(test_score)
    lista_steps = [step for step in steps]
    modelos_testados["Steps"].append(lista_steps)





df_modelos = pd.DataFrame({"Model":modelos_testados["Modelos"], "Score":modelos_testados["Score"], "Steps":modelos_testados["Steps"]})
df_modelos.to_markdown("Modelos.md",index=False)


#Salvando submissão com Max Score dentre nossos pipelines
x_test = pd.read_csv("Dados/test.csv",index_col = 0)
predict_array = clf.predict(x_test)
predict_submission = pd.DataFrame({"PassengerId":x_test.index,"Survived":predict_array})

predict_submission.to_csv("Predições/Predict2.csv",index=False)

print(df_modelos.Score)
print("FINALIZADO!!!")




