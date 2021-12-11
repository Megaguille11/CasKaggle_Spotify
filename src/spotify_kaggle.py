from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
%matplotlib notebook
from matplotlib import pyplot as plt
%matplotlib inline 
import scipy.stats
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
import statistics as st
import time

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem dataset d'exemple
dataset = load_dataset('../data/SpotifyFeatures.csv')
data = dataset.values

x = dataset.drop(columns = ['acousticness'])
y = dataset['acousticness']

print("Dimensionalitat de la BBDD:", dataset.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)



print("Número de valors no existents per cada atribut de la BBDD:")
print(dataset.isnull().sum())



print("Exemple de 10 entrades aleatòries de la BBDD:")
dataset.sample(10)



print("Estadístiques dels atributs numèrics de la BD:")
dataset.describe()



dataset = dataset.drop(dataset[dataset['duration_ms'] > 900000].index)
dataset = dataset.drop(dataset[dataset['loudness'] > 0].index)
dataset = dataset.drop(dataset[dataset['popularity'] == 0].index)

x = dataset.drop(columns = ['acousticness'])
y = dataset['acousticness']

print("Dimensionalitat de la BBDD:", dataset.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)



claus = dataset['key'].unique()
valors_claus = [1, 6, 0, 5, 7, 4, 3, 2, 10, 8, 9, 11]

for i in range(len(claus)):
    dataset.loc[dataset['key'] == claus[i], 'key'] = valors_claus[i]
    
    
    
    
modes = dataset['mode'].unique()
valors_modes = [1, 0]

for i in range(len(modes)):
    dataset.loc[dataset['mode'] == modes[i], 'mode'] = valors_modes[i]
    
    
    
compas = dataset['time_signature'].unique()
valors_compas = [3, 4, 2, 1, 0]

for i in range(len(compas)):
    dataset.loc[dataset['time_signature'] == compas[i], 'time_signature'] = valors_compas[i]
    
    
    
index_dataset = dataset.drop(columns = ['genre', 'artist_name', 'track_name', 'track_id'])
normalitzat = (index_dataset - index_dataset.mean()) / index_dataset.std()
normalitzat.head()



dataset['acousticness'].hist(bins=10)
plt.xlabel("Acousticness", fontsize=10)
plt.ylabel("Freqüència",fontsize=10)



correlacio = dataset.corr()

plt.figure(figsize=(13,13))

sns.heatmap(correlacio, annot=True, cmap="YlGnBu")



sns.pairplot(dataset, kind="hist")



dataset.loc[dataset['acousticness'] >= 0.5, 'acousticness'] = 1
dataset.loc[dataset['acousticness'] < 0.5, 'acousticness'] = 0

normalitzat['acousticness'] = dataset['acousticness']

subset = dataset[dataset['acousticness'] == 1]

sns.displot(subset['loudness'])
plt.title('Loudness de cançons amb una acústica més gran de 0.5')
sns.displot(dataset['loudness'])
plt.title('Loudness de totes les cançons de la BBDD')



training_data, testing_data = train_test_split(normalitzat, test_size=0.2, random_state=25)
y_trainVal = training_data['acousticness']
x_trainVal = training_data.drop(columns = ['acousticness'])
y_test = testing_data['acousticness']
x_test = testing_data.drop(columns = ['acousticness'])



n_models = 5
LR_model_def = LogisticRegression()
DT_model_def = DecisionTreeClassifier()
RF_model_def = RandomForestClassifier()
KNN_model_def = KNeighborsClassifier()
LR_max_val = 0
DT_max_val = 0
RF_max_val = 0
KNN_max_val = 0
LR_scores = []
DT_scores = []
RF_scores = []
KNN_scores = []
LR_time = 0
DT_time = 0
RF_time = 0
KNN_time = 0

for i in range(n_models):
    seed = 22*2*i
    x_train, x_val, y_train, y_val = train_test_split(x_trainVal, y_trainVal, test_size=0.2, random_state=seed)
    
    start = time.time()
    
    LR_model = LogisticRegression()
    LR_model.fit(x_train, y_train)
    LR_score = LR_model.score(x_val, y_val)
    LR_scores.append(LR_score)
    
    if LR_score > LR_max_val:
        LR_max_val = LR_score
        LR_model_def = LR_model
        
    end = time.time()
    LR_time = LR_time + (end - start)
       
    start = time.time()
    
    DT_model = DecisionTreeClassifier()
    DT_model.fit(x_train, y_train)
    DT_score = DT_model.score(x_val, y_val)
    DT_scores.append(DT_score)
    
    if DT_score > DT_max_val:
        DT_max_val = DT_score
        DT_model_def = DT_model
        
    end = time.time()
    DT_time = DT_time + (end - start)
        
        
    start = time.time()
        
    RF_model = RandomForestClassifier()
    RF_model.fit(x_train, y_train)
    RF_score = RF_model.score(x_val, y_val)
    RF_scores.append(RF_score)
    
    if RF_score > RF_max_val:
        RF_max_val = RF_score
        RF_model_def = RF_model
    
    end = time.time()
    RF_time = RF_time + (end - start)
    
    
    start = time.time()
    
    KNN_model = KNeighborsClassifier()
    KNN_model.fit(x_train, y_train)
    KNN_score = KNN_model.score(x_val, y_val)
    KNN_scores.append(KNN_score)
    
    if KNN_score > KNN_max_val:
        KNN_max_val = KNN_score
        KNN_model_def = KNN_model
             
    end = time.time()
    KNN_time = KNN_time + (end - start)
    
    
print("Precisió del nostre millor model amb el seu set de validació corresponent (Reg. Logística):", LR_max_val)
print("Desviació estàndard entre els diferents sets (Reg. Logística):", st.stdev(LR_scores))
print("Temps mitjà:", (LR_time / n_models), "segons.")
print(" ")
print("Precisió del nostre millor model amb el seu set de validació corresponent (Decision Tree):", DT_max_val)
print("Desviació estàndard entre els diferents sets (Decision Tree):", st.stdev(DT_scores))
print("Temps mitjà:", (DT_time / n_models), "segons.")
print(" ")
print("Precisió del nostre millor model amb el seu set de validació corresponent (Random Forest):", RF_max_val)
print("Desviació estàndard entre els diferents sets (Random Forest):", st.stdev(RF_scores))
print("Temps mitjà:", (RF_time / n_models), "segons.")
print(" ")
print("Precisió del nostre millor model amb el seu set de validació corresponent (K-NN):", KNN_max_val)
print("Desviació estàndard entre els diferents sets (K-NN):", st.stdev(KNN_scores))
print("Temps mitjà:", (KNN_time / n_models), "segons.")



print("Resulatats del Regressor Logístic:")
print("Classes:", LR_model_def.classes_)
print("Bias:", LR_model_def.intercept_[0])
print("Pesos dels atributs:")

i = 0
for col in x_val.columns:
    print(" ", col, ":",  LR_model_def.coef_[0][i])
    i += 1
    
    
    
LR_test_predictions = LR_model_def.predict(x_test)
LR_test_score = LR_model_def.score(x_test, y_test)
LR_matriu = confusion_matrix(y_test, LR_test_predictions)

print("Accuracy del set de test (Reg. Logística):", LR_test_score)
print(" ")
print("Matriu de confusió:")
sns.heatmap(LR_matriu, annot=True, fmt='g')



DT_test_predictions = DT_model_def.predict(x_test)
DT_test_score = DT_model_def.score(x_test, y_test)
DT_matriu = confusion_matrix(y_test, DT_test_predictions)

print("Accuracy del set de test (Decision Tree):", DT_test_score)
print(" ")
print("Matriu de confusió:")
sns.heatmap(DT_matriu, annot=True, fmt='g')



RF_test_predictions = RF_model_def.predict(x_test)
RF_test_score = RF_model_def.score(x_test, y_test)
RF_matriu = confusion_matrix(y_test, RF_test_predictions)

print("Accuracy del set de test (Random Forest):", RF_test_score)
print(" ")
print("Matriu de confusió:")
sns.heatmap(RF_matriu, annot=True, fmt='g')



KNN_test_predictions = KNN_model_def.predict(x_test)
KNN_test_score = KNN_model_def.score(x_test, y_test)
KNN_matriu = confusion_matrix(y_test, KNN_test_predictions)

print("Accuracy del set de test (K-NN):", KNN_test_score)
print(" ")
print("Matriu de confusió:")
sns.heatmap(KNN_matriu, annot=True, fmt='g')



start = time.time()

RF_model_200 = RandomForestClassifier(n_estimators=200)
RF_model_200.fit(x_train, y_train)
RF_test_predictions = RF_model_200.predict(x_test)
RF_200_test_score = RF_model_200.score(x_test, y_test)

end = time.time()
RF_200_time = end - start

print("Accuracy del set de test (Random Forest):", RF_200_test_score)
print("Temps mitjà:", RF_200_time, "segons.")


start = time.time()

KNN_model_10 = KNeighborsClassifier(n_neighbors=10)
KNN_model_10.fit(x_train, y_train)
KNN_10_test_predictions = KNN_model_10.predict(x_test)
KNN_10_test_score = KNN_model_10.score(x_test, y_test)

end = time.time()
KNN_10_time = end - start

print("Accuracy del set de test (KNN):", KNN_10_test_score)
print("Temps mitjà:", KNN_10_time, "segons.")



