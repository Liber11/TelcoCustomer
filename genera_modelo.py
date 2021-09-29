import sys
import subprocess
import pkg_resources

required = {'pycaret'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

import sklearn
if sklearn.__version__ != '0.23.2':
    raise Exception('Por favor, utilice la versión 0.23.2 de sklearn. La versión instalada es: {}'.format(sklearn.__version__))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import *
import pickle

#%% Definimos constantes

INPUT_FILE = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

MODELO = 'gbc'

PARAMETROS_MODELO_GBC = {'criterion': 'friedman_mse',
                     'max_depth': 3,
                     'max_features': None,
                     'max_leaf_nodes': 50,
                     'n_estimators': 40}

PARAMETROS_MODELO_RF = {'bootstrap': True,
                        'class_weight': None,
                        'criterion': 'gini',
                        'max_depth': 16,
                        'max_features': 'sqrt',
                        'max_leaf_nodes': 50,
                        'n_estimators': 40}

PARAMETROS_MODELO_LR = {'C': 0.03770949460475914,
                        'max_iter': 10000,
                        'multi_class': 'ovr',
                        'penalty': 'l1',
                        'solver': 'liblinear'}

#%% Importamos el dataset
df_data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

#%% Preprocesamiento del dataset

# Se borran los clientes nuevos y se pasa la columna TotalCharges a formato numérico
df_data['TotalCharges'].replace(' ', np.nan, inplace=True)
df_data.dropna(subset=['TotalCharges'], inplace=True)
df_data['TotalCharges'] = pd.to_numeric(df_data["TotalCharges"], downcast="float")

# Se crea un nuevo feature 'NumCode' con el código numérico del ID alfanumérico
df_data['NumCode'] = df_data['customerID'].str.split("-",expand=True).iloc[:,0]

# Se crea un nuevo feature 'AmountOfServices' de cantidad de servicios contratados
HasInternet = df_data['InternetService'].ne('No').astype(int)
df_data['AmountOfServices'] = HasInternet + df_data[['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']].eq('Yes').sum(axis=1)

#%% Se separa el dataset en una porción para entrenar y validar el modelo, y otra porción para test.

df_train_val, df_test = train_test_split(df_data, train_size=0.8,test_size=0.2)
df_test.to_csv(path_or_buf='df_test.csv')

#%% Configuración del dataset con librería PyCaret

# La función setup separa entre train y test, y configura features y otros parámetros del dataset
s = setup(df_train_val, target = 'Churn', silent=True, categorical_features=(['NumCode']), numeric_features=(['AmountOfServices']), ignore_features=(['customerID']))

#%% Creación del modelo predictivo

modelo = create_model(MODELO)

if MODELO == 'gbc':
    modelo.set_params(**PARAMETROS_MODELO_GBC)
elif MODELO == 'rf':
    modelo.set_params(**PARAMETROS_MODELO_RF)
elif MODELO == 'lr':
    modelo.set_params(**PARAMETROS_MODELO_LR)
else:
    raise Exception('Por favor, especifique uno de los modelos provistos (gbc, rf o lr). El modelo pedido fue: {}'.format(MODELO))
    
if MODELO in ['gbc','rf']:
    modelo = ensemble_model(modelo, method = 'Bagging')

#%%  Guardado del modelo en archivo pickle

save_model(modelo, 'modeLo_predictivo', model_only=True)










