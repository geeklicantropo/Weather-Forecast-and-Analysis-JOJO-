import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path
import os
from typing import Dict, List, Optional, Union

class VisualizationManager:
    def __init__(self, logger, style='classic', output_dir='Scripts/climate_prediction/outputs/plots'):
        self.logger = logger
        self.output_dir = output_dir
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 8)
        self._setup_directories()
        self.set_style()
        
    def _setup_directories(self):
        """Create necessary output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        
    def set_style(self):
        """Set global style parameters for plots."""
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 300,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def save_plot(self, fig, name: str, formats: List[str] = ['png', 'pdf', 'html']):
        """Save plot in multiple formats."""
        try:
            base_path = f"Scripts/climate_prediction/outputs/plots/{name}"
            for fmt in formats:
                if fmt == 'html' and isinstance(fig, go.Figure):
                    fig.write_html(f"{base_path}.html")
                elif fmt in ['png', 'pdf']:
                    if isinstance(fig, go.Figure):
                        fig.write_image(f"{base_path}.{fmt}")
                    else:
                        plt.savefig(f"{base_path}.{fmt}", dpi=300, bbox_inches='tight')
            
            self.logger.log_info(f"Saved plot {name} in formats: {formats}")
        except Exception as e:
            self.logger.log_error(f"Error saving plot {name}: {str(e)}")
    
    def plot_predictions(self, true_values: pd.Series, predictions: Dict[str, pd.DataFrame]):
        """Plot model predictions against true values."""
        try:
            # Static matplotlib version
            plt.figure(figsize=(15, 8))
            plt.plot(true_values.index, true_values.values, label='Actual', color='black', linewidth=2)
            
            for i, (model_name, pred) in enumerate(predictions.items()):
                pred_values = pred['forecast'] if isinstance(pred, pd.DataFrame) else pred
                plt.plot(true_values.index, pred_values, 
                        label=f'{model_name}', color=self.colors[i], alpha=0.7)
                
                if isinstance(pred, pd.DataFrame) and 'lower_bound' in pred.columns:
                    plt.fill_between(true_values.index, 
                                   pred['lower_bound'], pred['upper_bound'],
                                   color=self.colors[i], alpha=0.2)
            
            plt.title('Model Predictions Comparison')
            plt.xlabel('Date')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            self.save_plot(plt.gcf(), 'predictions_comparison_static')
            plt.close()
            
            # Interactive plotly version
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=true_values.index, y=true_values, 
                                   name='Actual', line=dict(color='black', width=2)))
            
            for i, (model_name, pred) in enumerate(predictions.items()):
                pred_values = pred['forecast'] if isinstance(pred, pd.DataFrame) else pred
                fig.add_trace(go.Scatter(x=true_values.index, y=pred_values,
                                       name=model_name))
                
                if isinstance(pred, pd.DataFrame) and 'lower_bound' in pred.columns:
                    fig.add_trace(go.Scatter(
                        x=true_values.index.tolist() + true_values.index.tolist()[::-1],
                        y=pred['upper_bound'].tolist() + pred['lower_bound'].tolist()[::-1],
                        fill='toself',
                        fillcolor=f'rgba{tuple(list(plt.cm.colors.to_rgb(self.colors[i])) + [0.2])}',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{model_name} CI'
                    ))
            
            fig.update_layout(
                title='Model Predictions Comparison (Interactive)',
                xaxis_title='Date',
                yaxis_title='Temperature (°C)',
                hovermode='x unified',
                template='plotly_white'
            )
            self.save_plot(fig, 'predictions_comparison_interactive')
            
        except Exception as e:
            self.logger.log_error(f"Error plotting predictions: {str(e)}")
    
    def plot_components(self, data: pd.Series, period: int = 24*7):
        """Plot time series decomposition."""
        try:
            decomposition = seasonal_decompose(data, period=period)
            
            # Static matplotlib version
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 15))
            
            # Observed
            decomposition.observed.plot(ax=ax1)
            ax1.set_title('Observed')
            
            # Trend
            decomposition.trend.plot(ax=ax2)
            ax2.set_title('Trend')
            
            # Seasonal
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title('Seasonal')
            
            # Residual
            decomposition.resid.plot(ax=ax4)
            ax4.set_title('Residual')
            
            plt.tight_layout()
            self.save_plot(plt.gcf(), 'time_series_decomposition')
            plt.close()
            
            # Interactive plotly version
            fig = make_subplots(rows=4, cols=1, 
                              subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'])
            
            components = [decomposition.observed, decomposition.trend, 
                         decomposition.seasonal, decomposition.resid]
            
            for i, component in enumerate(components, 1):
                fig.add_trace(go.Scatter(x=component.index, y=component.values),
                            row=i, col=1)
            
            fig.update_layout(height=1000, showlegend=False, title='Time Series Decomposition')
            self.save_plot(fig, 'time_series_decomposition_interactive')
            
        except Exception as e:
            self.logger.log_error(f"Error plotting components: {str(e)}")
    
    def plot_metrics_comparison(self, metrics: pd.DataFrame):
        """Plot comparison of model metrics."""
        try:
            # Static matplotlib version
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            metrics = metrics.transpose()
            
            for (metric, values), ax in zip(metrics.items(), axes.flatten()):
                values.plot(kind='bar', ax=ax)
                ax.set_title(f'{metric.upper()}')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            self.save_plot(plt.gcf(), 'metrics_comparison')
            plt.close()
            
            # Interactive plotly version
            fig = make_subplots(rows=2, cols=2, subplot_titles=list(metrics.items()))
            
            for i, (metric, values) in enumerate(metrics.items()):
                row, col = (i // 2) + 1, (i % 2) + 1
                fig.add_trace(
                    go.Bar(x=values.index, y=values.values, name=metric),
                    row=row, col=col
                )
            
            fig.update_layout(height=800, showlegend=False, title='Model Metrics Comparison')
            self.save_plot(fig, 'metrics_comparison_interactive')
            
        except Exception as e:
            self.logger.log_error(f"Error plotting metrics comparison: {str(e)}")
    
    def plot_forecast_horizon(self, historical: pd.Series, forecasts: Dict[str, pd.DataFrame],
                            forecast_start: pd.Timestamp):
        """Plot future forecasts with confidence intervals."""
        try:
            # Static matplotlib version
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
            
            # Interactive plotly version
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=historical.index, y=historical,
                                   name='Historical', line=dict(color='black', width=2)))
            
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
            
            fig.update_layout(
                title='Temperature Forecast (Interactive)',
                xaxis_title='Date',
                yaxis_title='Temperature (°C)',
                hovermode='x unified',
                template='plotly_white'
            )
            self.save_plot(fig, 'forecast_horizon_interactive')
            
        except Exception as e:
            self.logger.log_error(f"Error plotting forecast horizon: {str(e)}")
    
    def plot_feature_importance(self, feature_importance: Dict[str, pd.Series]):
        """Plot feature importance for each model."""
        try:
            n_models = len(feature_importance)
            fig = plt.figure(figsize=(15, 5 * n_models))
            
            for i, (model_name, importance) in enumerate(feature_importance.items()):
                ax = plt.subplot(n_models, 1, i+1)
                importance.sort_values().plot(kind='barh', ax=ax)
                ax.set_title(f'{model_name} Feature Importance')
            
            plt.tight_layout()
            self.save_plot(plt.gcf(), 'feature_importance')
            plt.close()
            
            # Interactive plotly version
            fig = make_subplots(rows=n_models, cols=1,
                              subplot_titles=[f'{model} Feature Importance' 
                                            for model in feature_importance.keys()])
            
            for i, (model_name, importance) in enumerate(feature_importance.items(), 1):
                importance = importance.sort_values()
                fig.add_trace(
                    go.Bar(x=importance.values, y=importance.index, orientation='h',
                          name=model_name),
                    row=i, col=1
                )
            
            fig.update_layout(height=400*n_models, showlegend=True,
                            title='Feature Importance by Model')
            self.save_plot(fig, 'feature_importance_interactive')
            
        except Exception as e:
            self.logger.log_error(f"Error plotting feature importance: {str(e)}")

if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    viz_manager = VisualizationManager(logger)
