from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base_model import BaseModel

class SARIMAModel(BaseModel):
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        super().__init__(name="SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
    
    def preprocess_data(self, df):
        return df[self.target_variable]
    
    def train(self, train_data):
        train_series = self.preprocess_data(train_data)
        self.model = SARIMAX(
            train_series,
            order=self.order,
            seasonal_order=self.seasonal_order
        )
        self.model = self.model.fit(disp=False)
    
    def predict(self, test_data):
        return self.model.forecast(steps=len(test_data))