# src/visualization/visualization_manager.py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

class VisualizationManager:
    def __init__(self, logger, style='seaborn-whitegrid', output_dir='outputs/plots'):
        self.logger = logger
        self.output_dir = output_dir
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 8)
        self.set_style()
    
    def set_style(self):
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 300
        })
    
    def save_plot(self, fig, name: str, formats: List[str] = ['png', 'html']):
        base_path = f"{self.output_dir}/{name}"
        for fmt in formats:
            if fmt == 'html' and isinstance(fig, go.Figure):
                fig.write_html(f"{base_path}.html")
            elif fmt == 'png':
                if isinstance(fig, go.Figure):
                    fig.write_image(f"{base_path}.png")
                else:
                    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')

    def plot_predictions(self, true_values: pd.Series, predictions: Dict[str, Union[np.ndarray, pd.DataFrame]], 
                        confidence_intervals: Optional[Dict] = None):
        # Static matplotlib version
        plt.figure(figsize=(15, 8))
        plt.plot(true_values.index, true_values.values, label='Actual', color='black', linewidth=2)
        
        for i, (model_name, preds) in enumerate(predictions.items()):
            preds_values = preds['forecast'] if isinstance(preds, pd.DataFrame) else preds
            plt.plot(true_values.index, preds_values, label=f'{model_name}', color=self.colors[i], alpha=0.7)
        
        plt.title('Model Predictions Comparison', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        self.save_plot(plt.gcf(), 'predictions_static')
        plt.close()
        
        # Interactive Plotly version
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=true_values.index, y=true_values, name='Actual', line=dict(color='black', width=2)))
        
        for i, (model_name, preds) in enumerate(predictions.items()):
            if isinstance(preds, pd.DataFrame):
                fig.add_trace(go.Scatter(x=true_values.index, y=preds['forecast'], name=model_name))
                fig.add_trace(go.Scatter(
                    x=true_values.index.tolist() + true_values.index.tolist()[::-1],
                    y=preds['upper_bound'].tolist() + preds['lower_bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor=f'rgba{tuple(list(plt.cm.colors.to_rgb(self.colors[i])) + [0.2])}',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{model_name} CI'
                ))
            else:
                fig.add_trace(go.Scatter(x=true_values.index, y=preds, name=model_name))
                
        fig.update_layout(
            title='Model Predictions Comparison (Interactive)',
            xaxis_title='Date',
            yaxis_title='Temperature (°C)',
            hovermode='x unified',
            template='plotly_white'
        )
        self.save_plot(fig, 'predictions_interactive')

    def plot_metrics_comparison(self, metrics: pd.DataFrame):
        # Enhanced static version with additional metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics = metrics.transpose()
        
        for (metric, values), ax in zip(metrics.items(), axes.flatten()):
            values.plot(kind='bar', ax=ax)
            ax.set_title(f'{metric.upper()} by Model')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.save_plot(plt.gcf(), 'metrics_comparison')
        plt.close()
        
        # Interactive version
        fig = make_subplots(rows=2, cols=2, subplot_titles=list(metrics.items()))
        
        for i, (metric, values) in enumerate(metrics.items()):
            row, col = (i // 2) + 1, (i % 2) + 1
            fig.add_trace(
                go.Bar(x=values.index, y=values.values, name=metric),
                row=row, col=col
            )
            
        fig.update_layout(height=800, showlegend=False)
        self.save_plot(fig, 'metrics_comparison_interactive')

    def plot_residuals_analysis(self, residuals: Dict[str, np.ndarray]):
        # Extended static version
        fig = plt.figure(figsize=(20, 6*len(residuals)))
        gs = plt.GridSpec(len(residuals), 4)
        
        for i, (model_name, resids) in enumerate(residuals.items()):
            # Distribution
            ax1 = plt.subplot(gs[i, 0])
            sns.histplot(resids, kde=True, ax=ax1)
            ax1.set_title(f'{model_name} Distribution')
            
            # Q-Q plot
            ax2 = plt.subplot(gs[i, 1])
            stats.probplot(resids, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot')
            
            # Time series
            ax3 = plt.subplot(gs[i, 2])
            ax3.plot(resids)
            ax3.set_title('Residuals over Time')
            
            # Autocorrelation
            ax4 = plt.subplot(gs[i, 3])
            pd.Series(resids).autocorr(lag=range(20)).plot(ax=ax4)
            ax4.set_title('Autocorrelation')
            
        plt.tight_layout()
        self.save_plot(plt.gcf(), 'residuals_analysis')
        plt.close()

    def plot_forecast_horizon(self, historical: pd.Series, forecasts: Dict[str, pd.DataFrame], 
                            forecast_start: pd.Timestamp):
        # Static version
        plt.figure(figsize=(15, 8))
        plt.plot(historical.index, historical, label='Historical', color='black', linewidth=2)
        
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            plt.plot(forecast.index, forecast['forecast'], 
                    label=f'{model_name} Forecast', color=self.colors[i])
            plt.fill_between(forecast.index, forecast['lower_bound'], forecast['upper_bound'],
                          alpha=0.2, color=self.colors[i])
        
        plt.axvline(x=forecast_start, color='gray', linestyle='--', label='Forecast Start')
        plt.title('Temperature Forecast with Confidence Intervals')
        plt.legend()
        self.save_plot(plt.gcf(), 'forecast_horizon')
        plt.close()
        
        # Interactive version
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical.index, y=historical, name='Historical', 
                                line=dict(color='black', width=2)))
        
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast['forecast'], 
                                   name=f'{model_name} Forecast'))
            fig.add_trace(go.Scatter(
                x=forecast.index.tolist() + forecast.index.tolist()[::-1],
                y=forecast['upper_bound'].tolist() + forecast['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(list(plt.cm.colors.to_rgb(self.colors[i])) + [0.2])}',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{model_name} CI'
            ))
            
        fig.add_vline(x=forecast_start, line_dash="dash", line_color="gray")
        fig.update_layout(title='Temperature Forecast (Interactive)', hovermode='x unified')
        self.save_plot(fig, 'forecast_horizon_interactive')

    def plot_components(self, data: pd.Series, period: int = 24*7):
        decomposition = seasonal_decompose(data, period=period)
        
        # Static version
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 15))
        components = ['Observed', 'Trend', 'Seasonal', 'Residual']
        series = [decomposition.observed, decomposition.trend, 
                 decomposition.seasonal, decomposition.resid]
        
        for ax, title, series in zip([ax1, ax2, ax3, ax4], components, series):
            series.plot(ax=ax)
            ax.set_title(title)
            
        plt.tight_layout()
        self.save_plot(plt.gcf(), 'decomposition')
        plt.close()
        
        # Interactive version
        fig = make_subplots(rows=4, cols=1, subplot_titles=components)
        
        for i, series in enumerate(series, 1):
            fig.add_trace(go.Scatter(x=series.index, y=series), row=i, col=1)
            
        fig.update_layout(height=1000, showlegend=False)
        self.save_plot(fig, 'decomposition_interactive')