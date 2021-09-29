import numpy as np
import pandas as pd
from pycaret.classification import *
import pickle

#%% Definimos constantes
INPUT_FILE = 'df_test.csv'

#%% Inputs del script

#Se importa el dataset a predecir
df_test = pd.read_csv('df_test.csv')

#% Se importa el modelo en archivo pickle y se lo aplica a los datos que se quieren predecir
modelo =  load_model('modelo_predictivo')

#%% Resultados

# Creado y guardado de dataframe con predicciones
df_prediccion = predict_model(modelo, data = df_test)
df_prediccion[['customerID','Churn','Label', 'MonthlyCharges', 'tenure']].to_csv('prediccion.csv')

# Matriz de confusión
plot_model(modelo, plot='error', save=True)

#Se realiza un mapa de color para las métricas del modelo
plot_model(modelo, plot='class_report', save=True)
