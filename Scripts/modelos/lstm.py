import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('seu_arquivo.csv', parse_dates=['data'], index_col='data')
scaler = MinMaxScaler(feature_range=(0, 1))
dados_normalizados = scaler.fit_transform(df.values)
tamanho_treino = int(len(dados_normalizados) * 0.8)
treino = dados_normalizados[:tamanho_treino]
teste = dados_normalizados[tamanho_treino:]

def criar_sequencias(dados, passos):
    X, y = [], []
    for i in range(len(dados) - passos):
        X.append(dados[i:(i + passos), 0])
        y.append(dados[i + passos, 0])
    return np.array(X), np.array(y)

passos = 10
X_treino, y_treino = criar_sequencias(treino, passos)
X_teste, y_teste = criar_sequencias(teste, passos)
X_treino = np.reshape(X_treino, (X_treino.shape[0], X_treino.shape[1], 1))
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

modelo = Sequential()
modelo.add(LSTM(50, return_sequences=True, input_shape=(passos, 1)))
modelo.add(LSTM(50))
modelo.add(Dense(1))
modelo.compile(optimizer='adam', loss='mean_squared_error')
modelo.fit(X_treino, y_treino, epochs=20, batch_size=32, validation_data=(X_teste, y_teste))

previsoes = modelo.predict(X_teste)
previsoes_invertidas = scaler.inverse_transform(previsoes)
y_teste_invertido = scaler.inverse_transform([y_teste])
rmse = np.sqrt(mean_squared_error(y_teste_invertido[0], previsoes_invertidas[:, 0]))
mae = mean_absolute_error(y_teste_invertido[0], previsoes_invertidas[:, 0])
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
