from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
%matplotlib notebook
from matplotlib import pyplot as plt
%matplotlib inline 
import scipy.stats
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split


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
dataset = dataset.drop(dataset[dataset['duration_ms'] > 900000].index)
dataset = dataset.drop(dataset[dataset['loudness'] > 0].index)
dataset = dataset.drop(dataset[dataset['popularity'] == 0].index)
dataset.describe()


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
    
    
dataset.sample(10)


index_dataset = dataset.drop(columns = ['genre', 'artist_name', 'track_name', 'track_id'])
normalitzat = (index_dataset - index_dataset.mean()) / index_dataset.std()
normalitzat.head()


dataset['acousticness'].hist(bins=10)
plt.xlabel("Acousticness", fontsize=10)
plt.ylabel("Freqüència",fontsize=10)


normalitzat.corr()


sns.pairplot(dataset)


subset = dataset[dataset['acousticness'] > 0.5]

sns.displot(subset['loudness'])
plt.title('Loudness de cançons amb una acústica més gran de 0.5')
sns.displot(dataset['loudness'])
plt.title('Loudness de totes les cançons de la BBDD')


dataset.loc[dataset['acousticness'] >= 0.5, 'acousticness'] = 1
dataset.loc[dataset['acousticness'] < 0.5, 'acousticness'] = 0

normalitzat['acousticness'] = dataset['acousticness']

normalitzat.sample(5)


training_data, testing_data = train_test_split(normalitzat, test_size=0.2, random_state=25)
y_trainVal = training_data['acousticness']
x_trainVal = training_data.drop(columns = ['acousticness'])
y_test = testing_data['acousticness']
x_test = testing_data.drop(columns = ['acousticness'])


n_models = 5
model_def = LogisticRegression()
max_val = 0

for i in range(n_models):
    seed = 22*2*i
    x_train, x_val, y_train, y_val = train_test_split(x_trainVal, y_trainVal, test_size=0.2, random_state=seed)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    preds = model.predict(x_val)
    score = model.score(x_val, y_val)
    
    if score > max_val:
        max_val = score
        model_def = model
        
print("Precisió del nostre millor model amb el seu set de validació corresponent:", max_val)


print("Classes:", model_def.classes_)
print("Bias:", model_def.intercept_[0])
print("Pesos dels atributs:")

i = 0
for col in x_val.columns:
    print(" ", col, ":",  model_def.coef_[0][i])
    i += 1
    
    
test_predictions = model.predict(x_test)
test_score = model.score(x_test, y_test)
matriu = confusion_matrix(y_test, test_predictions)

print("Matriu de confusió:")
print(" ", matriu[0])
print(" ", matriu[1])
print("Accuracy respecte el set de test:", test_score)
print("")

fpr, tpr, threshold = roc_curve(y_test, test_predictions)
roc_auc = auc(fpr, tpr)

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


