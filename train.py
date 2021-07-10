import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

#Model_Selection
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt
import seaborn as sns

from Pipeline import get_drop_categorical_features,get_drop_columns_with_null_valuse,get_colums_names
from Pipeline_mmansur import get_dealing_null_values, get_transform_dtype



df = pd.read_csv("Dados/train.csv",index_col=0)
x = df.drop("Survived",axis=1).copy()
y = df.Survived



x_train, x_val, y_train, y_val = train_test_split(x,y,
                                                    test_size = 0.3,
                                                    random_state = 0)


pipe1 = Pipeline(memory=None,
                 steps = [
                     ("Feature_Selection: get_drop_categorical_features",get_drop_categorical_features),
                     ("Null_Validate: get_drop_columns_with_null_valuse",get_drop_columns_with_null_valuse),
                     ("Final_Columns: get_colums_names",get_colums_names),
                     ("Model: RandomForest", RandomForestClassifier() )
                 ],
                verbose=False
                )

pipe2 = Pipeline(memory=None,
                 steps = [
                     ("Feature_Selection: get_drop_categorical_features",get_drop_categorical_features),
                     ("Null_Validate: get_drop_columns_with_null_valuse",get_drop_columns_with_null_valuse),
                     ("Final_Columns: get_colums_names",get_colums_names),
                     ("Model: LogisticRegression", LogisticRegression() )
                 ],
                verbose=False
                )

pipe3 = Pipeline(memory=None,
                 steps = [
                     ("Feature_Selection:get_transform_dtype",get_transform_dtype),
                     ("Null_Validate:get_dealing_null_values",get_dealing_null_values),
                     ("Final_Columns:get_colums_names",get_colums_names),
                     ("Model:RandomForest", RandomForestClassifier() )
                 ],
                verbose=False
                )

pipe4 = Pipeline(memory=None,
                 steps = [
                     ("Feature_Selection:get_transform_dtype",get_transform_dtype),
                     ("Null_Validate:get_dealing_null_values",get_dealing_null_values),
                     ("Final_Columns:get_colums_names",get_colums_names),
                     ("Model:LogisticRegression", LogisticRegression() )
                 ],
                verbose=False
                )
#Salvando Scores
modelos_testados = {"Modelos":["RandomForestClassifier","LogisticRegression","RandomForestClassifier_v2","LogisticRegression"],
                    "Pipeline":[pipe1,pipe2,pipe3, pipe4],
                    "Score":[],
                    "Steps":[]
                    }
n = len(modelos_testados["Modelos"])
with open("metrics.txt", 'w') as outfile:
    for ref in range(n):
        modelos_testados["Pipeline"][ref].fit(x_train,y_train)
        test_score = modelos_testados["Pipeline"][ref].score(x_val,y_val)
        nome_modelo = modelos_testados["Modelos"][ref]
        steps = modelos_testados["Pipeline"][ref].named_steps.keys()
        outfile.write(f"{nome_modelo}- Test Score: {test_score} - Steps: {steps}")
        modelos_testados["Score"].append(test_score)
        lista_steps = [step for step in steps]
        modelos_testados["Steps"].append(lista_steps)





df_modelos = pd.DataFrame({"Model":modelos_testados["Modelos"], "Score":modelos_testados["Score"], "Steps":modelos_testados["Steps"]})
df_modelos.to_markdown("Modelos.md",index=False)




#Salvando submissão
#Aqui devemos escolher nosso Pipe que iremos utilizar para nosso modelo
x_test = pd.read_csv("Dados/test.csv",index_col = 0)
predict_array = pipe3.predict(x_test)
predict_submission = pd.DataFrame({"PassengerId":x_test.index,"Survived":predict_array})
predict_submission.to_csv("Predições/Predict2.csv",index=False)


#Plotando grafico
importances = pipe1.named_steps['RandomForest'].feature_importances_
colunas = ['Pclass', 'SibSp', 'Parch']
# Colunas vai vir da var global em get_colum_names
feature_df = pd.DataFrame(list(zip(colunas, importances)), columns = ["feature","importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False,)


axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs)
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120)
plt.close()

print("FINALIZADO!!!")


#cml-publish feature_importance.png --md >> report.md
#          cml-send-comment report.md