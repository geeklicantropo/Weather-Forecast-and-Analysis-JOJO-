import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('seu_arquivo.csv', parse_dates=['data'], index_col='data')
tamanho_treino = int(len(df) * 0.8)
treino = df.iloc[:tamanho_treino]
teste = df.iloc[tamanho_treino:]

modelo = SARIMAX(treino, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
resultado = modelo.fit(disp=False)
previsoes = resultado.forecast(steps=len(teste))

rmse = np.sqrt(mean_squared_error(teste, previsoes))
mae = mean_absolute_error(teste, previsoes)
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
