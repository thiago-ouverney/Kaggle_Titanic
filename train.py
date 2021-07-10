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

from Pipeline import get_drop_categorical_features,get_drop_columns_with_null_valuse,get_colums_names

df = pd.read_csv("Dados/train.csv",index_col=0)
x = df.drop("Survived",axis=1).copy()
y = df.Survived



x_train, x_val, y_train, y_val = train_test_split(x,y,
                                                    test_size = 0.3,
                                                    random_state = 0)


pipe1 = Pipeline(memory=None,
                 steps = [
                     ("Feature_Selection",get_drop_categorical_features),
                     ("Null_Validate",get_drop_columns_with_null_valuse),
                     ("Final_Columns",get_colums_names),
                     ("RandomForest", RandomForestClassifier() )
                 ],
                verbose=False
                )



#Salvando submissão
x_test = pd.read_csv("Dados/test.csv",index_col = 0)
predict_array = pipe1.predict(x_test)
predict_submission = pd.DataFrame({"PassengerId":x_test.index,"Survived":predict_array})
predict_submission.to_csv("Predições/Predict1.csv",index=False)

#Salvando Scores
pipe1.fit(x_train,y_train)
test_score = pipe1.score(x_val,y_val)
with open("metrics.txt", 'w') as outfile:
    outfile.write(f"Test Score: {test_score}")

#Plotando grafico
importances = pipe1.named_steps['RandomForest'].feature_importances_
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

