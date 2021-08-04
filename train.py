import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split,  cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

import Pipeline

################# PARAMETROS #################
RF_params = {
    'RandomForest__criterion': ["gini"],
    'RandomForest__bootstrap': [True,False],
    'RandomForest__min_samples_leaf': [2,3,5, 10],
    'RandomForest__max_depth': [3,8,10,15, 25, 30],
    'RandomForest__n_estimators': [100,200,300, 500]
}
RF_best_params = {
  'RandomForest__bootstrap': [False],
 'RandomForest__criterion': ['gini'],
 'RandomForest__max_depth': [15],
 'RandomForest__min_samples_leaf': [3],
 'RandomForest__n_estimators': [200]}

XGBoost_params = {
    'XGBoost__learning_rate': [0.01,0.001,0.005]

}
################# DATASET #################

df = pd.read_csv("Dados/train.csv",index_col=0)
x = df.drop("Survived",axis=1).copy()
y = df.Survived
seed = 0




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
        cv = KFold(n_splits=(folds))
        scores = cross_val_score(pipe_analise, X_train, y_train, cv=cv)
        validation_score = sum(scores)/len(scores)

        modelos_testados["Params"].append("To Do")
        steps = pipe_analise.named_steps.keys()
        pipe_analise.fit(X_train,y_train)

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
pipe_1 = saving_predict(x,y,folds= 5, seed = seed, pipe = Pipeline.pipe_RF, nome_modelo= "Random Forest Cross Validation Baseline")
pipe_2 = saving_predict(x,y,folds= 5, seed = seed, pipe = Pipeline.pipe_GB, nome_modelo= "Gradient Boosting Cross Validation Baseline")
pipe_3 = saving_predict(x,y,folds= 5, seed = seed, pipe = Pipeline.pipe_XGBoost, nome_modelo= "XGBoost Cross Validation Baseline")


#pipe_3 = saving_predict(x,y,folds= 5, seed = seed, pipe = Pipeline.pipe_RF, grid_params= RF_best_params, nome_modelo = "Random Forest Best Params")

#Saving Final Dataframe
df_modelos = pd.DataFrame(modelos_testados)
df_modelos.to_markdown("Modelos.md",index=False)


#Salvando submissão
def saving_prediction(pipe,x,local = "Predições/Predict4.csv"):
    x_test = x.copy()
    predict_array = pipe.predict(x_test)
    predict_submission = pd.DataFrame({"PassengerId":x_test.index,"Survived":predict_array})
    predict_submission.to_csv(local,index=False)

saving_prediction(pipe_3,x_test)



