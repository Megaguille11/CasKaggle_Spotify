from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
%matplotlib notebook
from matplotlib import pyplot as plt
%matplotlib inline 
import scipy.stats
import seaborn as sns

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem dataset d'exemple
dataset = load_dataset('../data/SpotifyFeatures.csv')
data = dataset.values

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
    
    
compas = dataset['time_signature'].unique()
valors_compas = [3, 4, 2, 1, 0]

for i in range(len(compas)):
    dataset.loc[dataset['time_signature'] == compas[i], 'time_signature'] = valors_compas[i]
   
    
index_dataset = dataset.drop(columns = ['genre', 'artist_name', 'track_name', 'track_id'])
normalitzat = (index_dataset - index_dataset.mean()) / index_dataset.std()
normalitzat.head()


dataset['popularity'].hist(bins=100)
plt.xlabel("Popularitat", fontsize=10)
plt.ylabel("Freqüència",fontsize=10)


normalitzat.corr()


sns.pairplot(dataset)


sns.scatterplot(data=dataset, x="loudness", y="popularity")


from sklearn.model_selection import train_test_split

training_data, testing_data = train_test_split(normalitzat, test_size=0.2, random_state=25)
y_train = training_data['popularity']
x_train = training_data.drop(columns = ['popularity'])
y_test = testing_data['popularity']
x_test = testing_data.drop(columns = ['popularity'])