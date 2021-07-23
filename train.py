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
                     ("ModelRF", RandomForestClassifier(random_state=seed) )
                 ],
                verbose=False
                )

RF_params = {
    'RandomForest__criterion': ["gini", "entropy"],
    'RandomForest__bootstrap': [True],
    'RandomForest__min_samples_leaf': [3, 4, 5],
    'RandomForest__n_estimators': [1000]
}


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
modelos_testados = {"Modelos":["RandomForestClassifier","LogisticRegression"],
                    "Pipeline":[pipe_RF,pipe_LR],
                    "Score":[],
                    "Steps":[],
                    "Params":[]
                    }


n = len(modelos_testados["Modelos"])
with open("metrics.txt", 'w') as outfile:
    for n in range(n):
        if str(modelos_testados["Pipeline"][n]).startswith("RandomForestClassifier"):
            pipe = modelos_testados["Pipeline"][n].fit(x_train,y_train)
            clf = GridSearchCV(pipe, RF_params)
            clf.fit(x_train, y_train)
            test_score = clf.score(x_val, y_val)
            steps = clf.estimator.named_steps.keys()
            params = clf.best_params_

        else:
            modelos_testados["Pipeline"][n].fit(x_train, y_train)
            test_score = modelos_testados["Pipeline"][n].score(x_val,y_val)
            steps = modelos_testados["Pipeline"][n].named_steps.keys()

        nome_modelo = modelos_testados["Modelos"][n]
        outfile.write(f"{nome_modelo}- Test Score: {test_score} - Steps: {steps}")

        modelos_testados["Score"].append(test_score)
        lista_steps = [step for step in steps]
        modelos_testados["Steps"].append(lista_steps)





df_modelos = pd.DataFrame({"Model":modelos_testados["Modelos"], "Score":modelos_testados["Score"], "Steps":modelos_testados["Steps"]})
df_modelos.to_markdown("Modelos.md",index=False)


#Salvando submissão com Max Score dentre nossos pipelines
index_max = modelos_testados["Score"].index(max(modelos_testados["Score"]))
best_pipe = modelos_testados["Pipeline"][index_max]
best_pipe_name = modelos_testados["Modelos"][index_max]

x_test = pd.read_csv("Dados/test.csv",index_col = 0)
predict_array = best_pipe.predict(x_test)
predict_submission = pd.DataFrame({"PassengerId":x_test.index,"Survived":predict_array})

predict_submission.to_csv("Predições/Predict2.csv",index=False)

print(df_modelos.head())
print("FINALIZADO!!!")




