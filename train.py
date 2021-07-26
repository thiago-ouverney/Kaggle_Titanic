import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split,  cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

from Pipeline import *



df = pd.read_csv("Dados/train.csv",index_col=0)
x = df.drop("Survived",axis=1).copy()
y = df.Survived
seed = 0


pipe_RF = Pipeline(memory=None,
                 steps = [
                     ("Dtype_Columns",get_transform_dtype_new),
                     ("Null_Validate:get_dealing_null_values",get_dealing_null_values),
                     ("RandomForest", RandomForestClassifier(random_state=seed) )
                 ],
                verbose=False
                )

RF_params = {
    'Dtype_Columns__kw_args': [ {'teste': "ok"} ],
    'RandomForest__criterion': ["gini"],
    'RandomForest__bootstrap': [True,False],
    'RandomForest__min_samples_leaf': [3,5, 10],
    'RandomForest__max_depth': [15,20, 25,27, 30],
    'RandomForest__n_estimators': [200, 500,700]
}
RF_best_params = {
 'Dtype_Columns__kw_args': [ {'teste': "ok"} ],  #Aqui conseguimos colocar usando __kw_args, argumentos proprios
  'RandomForest__bootstrap': [False],
 'RandomForest__criterion': ['gini'],
 'RandomForest__max_depth': [15],
 'RandomForest__min_samples_leaf': [3],
 'RandomForest__n_estimators': [200]}

#Salvando Scores
modelos_testados = {"Modelos":[],
                    "Score_Validation":[],
                    "Score_Test":[],
                    "Steps":[],
                    "Params":[]
                    }


# Predição para Baseline
def saving_predict(X,y,folds,seed,pipe,nome_modelo,grid_params = ""):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 / folds), random_state=seed)

    if grid_params:
        pipe_analise = GridSearchCV(pipe, grid_params, verbose=1,cv=folds)
        pipe_analise.fit(X_train, y_train)
        steps = pipe_analise.estimator.named_steps.keys()
        validation_score = pipe_analise.best_score_
        params = pipe_analise.best_params_
        modelos_testados["Params"].append(params)

    else:
        pipe_analise = pipe
        pipe_analise.fit(X_train,y_train)
        cv = KFold(n_splits=(folds))
        scores = cross_val_score(pipe, X_train, y_train, cv=cv)
        validation_score = sum(scores)/len(scores)

        modelos_testados["Params"].append("To Do")
        steps = pipe.named_steps.keys()


    #Saving test metric
    test_score = pipe_analise.score(X_test,y_test)
    lista_steps = [step for step in steps]

    modelos_testados["Modelos"].append(nome_modelo)
    modelos_testados["Steps"].append(lista_steps)
    modelos_testados["Score_Validation"].append(validation_score)
    modelos_testados["Score_Test"].append(test_score)
    return pipe_analise

# Teste
x_test = pd.read_csv("Dados/test.csv",index_col = 0)


# Saving my predictions
pipe_1 = saving_predict(x,y,folds= 5, seed = seed, pipe = pipe_RF,grid_params = RF_best_params, nome_modelo= "RF Parametros Otimizados")
#saving_predict(x,y,folds= 5, seed = seed, pipe = pipe_RF,grid_params = RF_params, nome_modelo= "RF Parametros a se otimizar")
pipe_2 = saving_predict(x,y,folds= 5, seed = seed, pipe = pipe_RF, nome_modelo= "RF Cross Validation Baseline")


#Saving Final Dataframe
df_modelos = pd.DataFrame(modelos_testados)
df_modelos.to_markdown("Modelos.md",index=False)


#Salvando submissão com Max Score dentre nossos pipelines

#predict_array = clf.predict(x_test)
#predict_submission = pd.DataFrame({"PassengerId":x_test.index,"Survived":predict_array})

#predict_submission.to_csv("Predições/Predict2.csv",index=False)





