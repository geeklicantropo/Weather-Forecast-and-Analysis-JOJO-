# src/visualization/model_visualizer.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List

class ModelVisualizer:
    def __init__(self, output_dir='outputs/plots'):
        self.output_dir = output_dir
        plt.style.use('seaborn')
        self.set_style()
    
    def set_style(self):
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 16,
            'figure.figsize': (12, 8),
            'figure.dpi': 300
        })

    def plot_model_comparison(self, comparison_results: pd.DataFrame):
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)
        
        # RMSE and MAE comparison
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['rmse', 'mae']
        comparison_results[metrics].plot(kind='bar', ax=ax1)
        ax1.set_title('Error Metrics Comparison')
        ax1.set_ylabel('Error Value')
        ax1.tick_params(axis='x', rotation=45)
        
        # R² Score comparison
        ax2 = fig.add_subplot(gs[0, 1])
        comparison_results['r2'].plot(kind='bar', ax=ax2)
        ax2.set_title('R² Score Comparison')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Prediction Interval Coverage
        ax3 = fig.add_subplot(gs[1, 0])
        comparison_results['prediction_interval_coverage'].plot(kind='bar', ax=ax3)
        ax3.set_title('Prediction Interval Coverage')
        ax3.set_ylabel('Coverage (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Interval Width
        ax4 = fig.add_subplot(gs[1, 1])
        comparison_results['mean_interval_width'].plot(kind='bar', ax=ax4)
        ax4.set_title('Mean Prediction Interval Width')
        ax4.set_ylabel('Width')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png')
        plt.close()

    def plot_predictions(self, true_values: pd.Series, predictions: Dict[str, pd.DataFrame], 
                        ensemble_predictions: np.ndarray = None):
        fig = go.Figure()
        
        # Plot true values
        fig.add_trace(go.Scatter(
            x=true_values.index,
            y=true_values,
            name='True Values',
            line=dict(color='black', width=2)
        ))
        
        colors = ['blue', 'red', 'green']
        for (model_name, pred), color in zip(predictions.items(), colors):
            # Plot predictions with confidence intervals
            fig.add_trace(go.Scatter(
                x=pred.index,
                y=pred['forecast'],
                name=f'{model_name} Predictions',
                line=dict(color=color)
            ))
            
            fig.add_trace(go.Scatter(
                x=pred.index.tolist() + pred.index.tolist()[::-1],
                y=pred['upper_bound'].tolist() + pred['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(list(plt.cm.colors.to_rgb(color)) + [0.2])}',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{model_name} Confidence Interval'
            ))
        
        if ensemble_predictions is not None:
            fig.add_trace(go.Scatter(
                x=pred.index,
                y=ensemble_predictions,
                name='Ensemble Predictions',
                line=dict(color='purple', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Model Predictions Comparison',
            xaxis_title='Date',
            yaxis_title='Temperature (°C)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.write_html(f'{self.output_dir}/predictions_comparison.html')

    def plot_residuals(self, true_values: pd.Series, predictions: Dict[str, pd.DataFrame]):
        fig = make_subplots(rows=2, cols=2, subplot_titles=[f'{model} Residuals' for model in predictions.keys()])
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            residuals = true_values - pred['forecast']
            row, col = (i // 2) + 1, (i % 2) + 1
            
            # Q-Q plot
            fig.add_trace(go.Scatter(
                x=np.sort(np.random.normal(0, 1, len(residuals))),
                y=np.sort(residuals),
                mode='markers',
                name=f'{model_name} Q-Q Plot'
            ), row=row, col=col)
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[-3, 3],
                y=[-3, 3],
                mode='lines',
                line=dict(dash='dash'),
                showlegend=False
            ), row=row, col=col)
        
        fig.update_layout(height=800, title_text="Residual Analysis", showlegend=True)
        fig.write_html(f'{self.output_dir}/residuals_analysis.html')

    def plot_feature_importance(self, feature_importance: Dict[str, pd.Series]):
        n_models = len(feature_importance)
        fig = plt.figure(figsize=(15, 5 * n_models))
        
        for i, (model_name, importance) in enumerate(feature_importance.items()):
            ax = plt.subplot(n_models, 1, i+1)
            importance.sort_values().plot(kind='barh', ax=ax)
            ax.set_title(f'{model_name} Feature Importance')
            
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png')
        plt.close()

    def plot_time_series_components(self, decomposition, model_name):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
        
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('Original Time Series')
        
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        
        decomposition.resid.plot(ax=ax4)
        ax4.set_title('Residuals')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{model_name}_decomposition.png')
        plt.close()

    def plot_forecast_horizon(self, true_values: pd.Series, predictions: Dict[str, pd.DataFrame],
                            forecast_start: pd.Timestamp):
        fig = go.Figure()
        
        # Plot historical data
        fig.add_trace(go.Scatter(
            x=true_values.index,
            y=true_values,
            name='Historical Data',
            line=dict(color='black', width=2)
        ))
        
        colors = ['blue', 'red', 'green']
        for (model_name, pred), color in zip(predictions.items(), colors):
            mask = pred.index >= forecast_start
            pred_future = pred[mask]
            
            # Plot future predictions with confidence intervals
            fig.add_trace(go.Scatter(
                x=pred_future.index,
                y=pred_future['forecast'],
                name=f'{model_name} Forecast',
                line=dict(color=color)
            ))
            
            fig.add_trace(go.Scatter(
                x=pred_future.index.tolist() + pred_future.index.tolist()[::-1],
                y=pred_future['upper_bound'].tolist() + pred_future['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(list(plt.cm.colors.to_rgb(color)) + [0.2])}',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{model_name} Confidence Interval'
            ))
        
        # Add vertical line at forecast start
        fig.add_vline(x=forecast_start, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title='Future Temperature Forecasts',
            xaxis_title='Date',
            yaxis_title='Temperature (°C)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.write_html(f'{self.output_dir}/forecast_horizon.html')