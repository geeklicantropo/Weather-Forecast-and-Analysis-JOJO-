import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_lightning import Trainer

# Carregar os dados
df = pd.read_csv("seu_arquivo_clima.csv")
df['data'] = pd.to_datetime(df['data'])  # Supondo que você tenha uma coluna de data
df['time_idx'] = (df['data'] - df['data'].min()).dt.days  # Índice de tempo

# Configuração do TimeSeriesDataSet
max_encoder_length = 30  # Dias de histórico para treinar
max_prediction_length = 7  # Dias futuros para prever

training_cutoff = df["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[df["time_idx"] <= training_cutoff],
    time_idx="time_idx",
    target="temperatura",  # Coluna alvo que representa o clima
    group_ids=["cidade"],  # Agrupar por cidade (ajustar conforme os dados)
    min_encoder_length=max_encoder_length // 2,  # Histórico mínimo
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["cidade"],  # Coluna de dados estáticos (ex: cidade)
    time_varying_known_categoricals=["mês"],  # Exemplo de variável conhecida
    time_varying_known_reals=["time_idx"],  # Variável contínua conhecida (tempo)
    time_varying_unknown_reals=["temperatura"],  # Alvo desconhecido
    target_normalizer=NaNLabelEncoder(),  # Normalização do alvo
)

# Configuração dos DataLoaders
validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# Definição e Treinamento do Modelo
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,  # Tamanho da camada oculta
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # Saída com 7 previsões (quantile outputs)
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

trainer = Trainer(max_epochs=30, gpus=1 if torch.cuda.is_available() else 0)
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Previsão
best_model_path = trainer.checkpoint_callback.best_model_path
tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
predictions, _ = tft.predict(val_dataloader, return_index=True, return_decoder_lengths=True)

# Visualizar Resultados
print(predictions)
