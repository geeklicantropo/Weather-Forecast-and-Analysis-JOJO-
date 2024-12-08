import os
import yaml
from pathlib import Path

def setup_config():
    project_root = Path(__file__).parent
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)
    
    model_config = {
        'training': {
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 100,
            'early_stopping_patience': 10,
            'validation_split': 0.2,
            'cross_validation': {
                'n_splits': 5,
                'shuffle': False
            },
            'optimizer': {
                'type': 'adam',
                'parameters': {
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'epsilon': 1e-8,
                    'weight_decay': 0.0001
                }
            },
            'scheduler': {
                'type': 'reduce_on_plateau',
                'parameters': {
                    'mode': 'min',
                    'factor': 0.5,
                    'patience': 5,
                    'min_lr': 1e-6
                }
            }
        },
        'feature_engineering': {
            'temporal_features': {
                'cyclical': ['hour', 'day', 'month', 'day_of_week'],
                'categorical': ['season', 'is_weekend', 'is_holiday'],
                'numerical': ['day_of_year', 'week_of_year']
            },
            'rolling_windows': {
                'short_term': [24, 48, 72],
                'medium_term': [168, 336, 504],
                'long_term': [720, 2160, 4320]
            },
            'aggregations': {
                'functions': ['mean', 'std', 'min', 'max', 'range', 'skew', 'kurt']
            },
            'interaction_features': [
                ["TEMPERATURA DO AR - BULBO SECO HORARIA °C", "UMIDADE RELATIVA DO AR HORARIA %"],
                ["PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB", "TEMPERATURA DO AR - BULBO SECO HORARIA °C"]
            ],
            'seasonal_decomposition': {
                'period': 24,
                'model': 'additive',
                'extrapolate_trend': 'freq'
            }
        },
        'preprocessing': {
            'target_variable': "TEMPERATURA DO AR - BULBO SECO HORARIA °C",
            'datetime_format': "%Y-%m-%d %H:%M:%S",
            'invalid_values': [-9999.0, -999.0, -99.0, 9999.0],
            'missing_values': {
                'strategy': {
                    'short_gaps': "forward_fill",
                    'medium_gaps': "interpolate",
                    'long_gaps': "seasonal_interpolate"
                },
                'interpolation_methods': {
                    'numeric': 'cubic',
                    'categorical': 'mode'
                },
                'interpolation_limit': 24
            },
            'outlier_detection': {
                'method': 'zscore',
                'threshold': 3.0,
                'rolling_window': 720
            },
            'scaling': {
                'method': 'robust',
                'parameters': {
                    'quantile_range': [0.01, 0.99]
                }
            }
        },
        'gpu': {
            'memory_fraction': 0.8,
            'precision': "float32",
            'batch_growth_rate': 1.5,
            'memory_growth': True,
            'allow_memory_growth': True,
            'per_process_gpu_memory_fraction': 0.9,
            'optimization': {
                'mixed_precision': True,
                'xla': True,
                'tensor_cores': True
            }
        },
        'models': {
            'lstm': {
                'enabled': True,
                'architecture': {
                    'input_size': None,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.1,
                    'bidirectional': True
                },
                'training': {
                    'sequence_length': 168,
                    'stride': 24,
                    'batch_norm': True,
                    'layer_norm': True,
                    'gradient_clip': 1.0
                }
            },
            'sarima': {
                'enabled': True,
                'order': {
                    'p': [0, 1, 2],
                    'd': [0, 1],
                    'q': [0, 1, 2]
                },
                'seasonal_order': {
                    'P': [0, 1],
                    'D': [0, 1],
                    'Q': [0, 1],
                    's': [24, 168, 720]
                },
                'fitting': {
                    'method': 'lbfgs',
                    'maxiter': 50,
                    'optim_score': 'aic',
                    'enforce_stationarity': True,
                    'enforce_invertibility': True
                }
            },
            'tft': {
                'enabled': True,
                'architecture': {
                    'hidden_size': 64,
                    'attention_head_size': 4,
                    'dropout': 0.1,
                    'hidden_continuous_size': 32,
                    'categorical_embedding_dim': 8
                },
                'training': {
                    'max_encoder_length': 720,
                    'min_encoder_length': 168,
                    'max_prediction_length': 168
                }
            }
        },
        'output': {
            'base_dir': 'outputs',
            'subdirs': {
                'models': 'models',
                'predictions': 'predictions',
                'plots': 'plots',
                'metrics': 'metrics',
                'logs': 'logs'
            },
            'forecast': {
                'horizon': 87600,
                'frequency': 'H',
                'include_history': True,
                'confidence_intervals': [0.1, 0.9]
            },
            'metrics': {
                'format': 'json',
                'include': [
                    'rmse', 'mae', 'mape', 'r2',
                    'coverage_rate', 'interval_score'
                ]
            },
            'visualization': {
                'style': 'classic',
                'context': 'paper',
                'dpi': 300,
                'formats': ['png', 'html', 'svg'],
                'figure_sizes': {
                    'default': [12, 8],
                    'wide': [15, 8],
                    'tall': [10, 12]
                },
                'fonts': {
                    'family': 'Arial',
                    'size': 12,
                    'title_size': 14
                },
                'color_palette': 'deep'
            },
            'logging': {
                'level': "INFO",
                'format': "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                'date_format': "%Y-%m-%d %H:%M:%S"
            }
        },
        'validation': {
            'test_size': 0.2,
            'validation_size': 0.1,
            'cross_validation': {
                'n_splits': 5,
                'gap': 24
            },
            'metrics_threshold': {
                'rmse': 2.0,
                'mae': 1.5,
                'mape': 10.0,
                'r2': 0.8
            },
            'data_quality': {
                'missing_threshold': 0.2,
                'correlation_threshold': 0.95,
                'variance_threshold': 0.01
            }
        }
    }
    
    with open(config_dir / "model_config.yaml", 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)

if __name__ == "__main__":
    setup_config()