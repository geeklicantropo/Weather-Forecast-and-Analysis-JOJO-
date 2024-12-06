import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
import logging
from datetime import datetime

class ConfigManager:
    def __init__(self, config_path: str = "./config/model_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._setup_paths()
        self._setup_logging()

    def _load_config(self) -> Dict[str, Any]:
        """Load and merge configuration files."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load environment-specific overrides if they exist
            env_config_path = os.getenv('MODEL_CONFIG_PATH')
            if env_config_path and os.path.exists(env_config_path):
                with open(env_config_path, 'r') as f:
                    env_config = yaml.safe_load(f)
                config = self._deep_update(config, env_config)
                
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {str(e)}")

    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                base_dict[key] = self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict

    def _validate_config(self):
        """Validate configuration parameters."""
        required_sections = ['training', 'feature_engineering', 'preprocessing', 'gpu', 'models', 'output', 'validation']
        
        # Check required sections
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate specific parameters
        self._validate_training_config()
        self._validate_model_configs()
        self._validate_output_config()

    def _validate_training_config(self):
        """Validate training configuration parameters."""
        training_config = self.config['training']
        
        # Validate batch size
        if training_config['batch_size'] <= 0:
            raise ValueError("Batch size must be positive")
            
        # Validate learning rate
        if not (0 < training_config['learning_rate'] < 1):
            raise ValueError("Learning rate must be between 0 and 1")
            
        # Validate epochs
        if training_config['epochs'] <= 0:
            raise ValueError("Number of epochs must be positive")

    def _validate_model_configs(self):
        """Validate model-specific configurations."""
        models_config = self.config['models']
        
        # Validate LSTM config
        lstm_config = models_config['lstm']
        if lstm_config['architecture']['hidden_size'] <= 0:
            raise ValueError("LSTM hidden size must be positive")
        if not (0 <= lstm_config['architecture']['dropout'] < 1):
            raise ValueError("LSTM dropout must be between 0 and 1")
            
        # Validate SARIMA config
        sarima_config = models_config['sarima']
        for param in ['p', 'd', 'q']:
            if not all(isinstance(x, int) and x >= 0 for x in sarima_config['order'][param]):
                raise ValueError(f"SARIMA {param} parameters must be non-negative integers")
                
        # Validate TFT config
        tft_config = models_config['tft']
        if tft_config['architecture']['hidden_size'] <= 0:
            raise ValueError("TFT hidden size must be positive")

    def _validate_output_config(self):
        """Validate output configuration parameters."""
        output_config = self.config['output']
        
        # Validate forecast horizon
        if output_config['forecast']['horizon'] <= 0:
            raise ValueError("Forecast horizon must be positive")
            
        # Validate confidence intervals
        intervals = output_config['forecast']['confidence_intervals']
        if not all(0 <= x <= 1 for x in intervals):
            raise ValueError("Confidence intervals must be between 0 and 1")

    def _setup_paths(self):
        """Create necessary directories based on configuration."""
        base_dir = self.config['output']['base_dir']
        for subdir in self.config['output']['subdirs'].values():
            path = os.path.join(base_dir, subdir)
            os.makedirs(path, exist_ok=True)

    def _setup_logging(self):
        """Configure logging based on configuration."""
        log_config = self.config['output']['logging']
        log_path = os.path.join(
            self.config['output']['base_dir'],
            self.config['output']['subdirs']['logs'],
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            datefmt=log_config['date_format'],
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )

    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration or specific section."""
        if section:
            if section not in self.config:
                raise KeyError(f"Configuration section '{section}' not found")
            return self.config[section]
        return self.config

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        self.config = self._deep_update(self.config, updates)
        self._validate_config()

    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model."""
        if model_name not in self.config['models']:
            raise KeyError(f"Configuration for model '{model_name}' not found")
        return self.config['models'][model_name]

    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.config['feature_engineering']

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.config['preprocessing']

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config['training']

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config['output']

    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self.config['validation']

    def get_gpu_config(self) -> Dict[str, Any]:
        """Get GPU configuration."""
        return self.config['gpu']

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        return self.config[key]

class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass