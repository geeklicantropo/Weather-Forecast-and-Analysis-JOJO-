# src/data_processing/data_validator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import logging
from scipy import stats
from datetime import datetime, timedelta
from src.utils.config_manager import ConfigManager

@dataclass
class ValidationReport:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict
    missing_data: Dict
    outliers: Dict
    data_quality_score: float

class DataValidator:
    def __init__(self, config_manager, logger):
        self.config = config_manager.get_validation_config()
        self.preprocessing_config = config_manager.get_preprocessing_config()
        self.logger = logger
        self.target_variable = self.preprocessing_config['target_variable']
        self.required_columns = self._get_required_columns()
        
    def _get_required_columns(self) -> Set[str]:
        """Define required columns based on configuration."""
        required = {
            self.target_variable,
            'PRECIPITACÃO TOTAL HORÁRIO MM',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB',
            'RADIACAO GLOBAL KJ/M²',
            'UMIDADE RELATIVA DO AR HORARIA %',
            'VENTO VELOCIDADE HORARIA M/S',
            'LATITUDE',
            'LONGITUDE',
            'ALTITUDE'
        }
        return required

    def validate_dataset(self, df: pd.DataFrame) -> ValidationReport:
        """Perform comprehensive dataset validation."""
        errors = []
        warnings = []
        statistics = {}
        
        try:
            # Schema validation
            schema_errors = self._validate_schema(df)
            errors.extend(schema_errors)
            
            # Data type validation
            dtype_errors = self._validate_datatypes(df)
            errors.extend(dtype_errors)
            
            # Time continuity validation
            continuity_errors, continuity_warnings = self._validate_time_continuity(df)
            errors.extend(continuity_errors)
            warnings.extend(continuity_warnings)
            
            # Missing data analysis
            missing_data = self._analyze_missing_data(df)
            if missing_data['total_missing_rate'] > self.config['data_quality']['missing_threshold']:
                errors.append(f"Missing data rate ({missing_data['total_missing_rate']:.2%}) exceeds threshold")
            
            # Outlier detection
            outliers = self._detect_outliers(df)
            
            # Calculate basic statistics
            statistics = self._calculate_statistics(df)
            
            # Data quality score
            quality_score = self._calculate_quality_score(df, missing_data, outliers)
            
            # Feature correlation analysis
            correlation_warnings = self._analyze_correlations(df)
            warnings.extend(correlation_warnings)
            
            # Range validation
            range_errors = self._validate_ranges(df)
            errors.extend(range_errors)
            
            validation_report = ValidationReport(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                statistics=statistics,
                missing_data=missing_data,
                outliers=outliers,
                data_quality_score=quality_score
            )
            
            self._log_validation_results(validation_report)
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise

    def _validate_schema(self, df: pd.DataFrame) -> List[str]:
        """Validate dataset schema."""
        errors = []
        missing_columns = self.required_columns - set(df.columns)
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            
        return errors

    def _validate_datatypes(self, df: pd.DataFrame) -> List[str]:
        """Validate column data types."""
        errors = []
        expected_types = {
            'DATETIME': 'datetime64[ns]',
            self.target_variable: 'float64',
            'PRECIPITACÃO TOTAL HORÁRIO MM': 'float64',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': 'float64',
            'LATITUDE': 'float64',
            'LONGITUDE': 'float64',
            'ALTITUDE': 'float64'
        }
        
        for column, expected_type in expected_types.items():
            if column in df.columns and str(df[column].dtype) != expected_type:
                errors.append(f"Invalid dtype for {column}: expected {expected_type}, got {df[column].dtype}")
                
        return errors

    def _validate_time_continuity(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate time series continuity."""
        errors = []
        warnings = []
        
        # Sort by datetime
        df = df.sort_index()
        
        # Check time step consistency
        time_diff = df.index.to_series().diff()
        expected_diff = pd.Timedelta(hours=1)
        
        # Find gaps and inconsistencies
        gaps = time_diff[time_diff > expected_diff]
        if not gaps.empty:
            for timestamp, gap in gaps.items():
                warning = f"Time gap of {gap} detected at {timestamp}"
                warnings.append(warning)
                
        # Check for duplicates
        duplicates = df.index.duplicated()
        if duplicates.any():
            errors.append(f"Found {duplicates.sum()} duplicate timestamps")
            
        return errors, warnings

    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict:
        """Analyze missing data patterns."""
        missing_stats = {
            'total_missing_rate': df.isnull().mean().mean(),
            'missing_by_column': df.isnull().mean().to_dict(),
            'missing_patterns': df.isnull().sum(axis=1).value_counts().to_dict(),
            'consecutive_missing': self._get_consecutive_missing(df)
        }
        
        return missing_stats

    def _get_consecutive_missing(self, df: pd.DataFrame) -> Dict:
        """Calculate consecutive missing values statistics."""
        consecutive_missing = {}
        for column in df.columns:
            mask = df[column].isnull()
            if mask.any():
                consecutive_lengths = []
                current_length = 0
                for value in mask:
                    if value:
                        current_length += 1
                    elif current_length > 0:
                        consecutive_lengths.append(current_length)
                        current_length = 0
                if current_length > 0:
                    consecutive_lengths.append(current_length)
                    
                consecutive_missing[column] = {
                    'max_consecutive': max(consecutive_lengths) if consecutive_lengths else 0,
                    'mean_consecutive': np.mean(consecutive_lengths) if consecutive_lengths else 0
                }
                
        return consecutive_missing

    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers using multiple methods."""
        outliers = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            iqr = stats.iqr(df[column].dropna())
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            
            outliers[column] = {
                'zscore_outliers': (z_scores > 3).sum(),
                'iqr_outliers': ((df[column] < (q1 - 1.5 * iqr)) | (df[column] > (q3 + 1.5 * iqr))).sum(),
                'percentile_outliers': ((df[column] < df[column].quantile(0.01)) | 
                                      (df[column] > df[column].quantile(0.99))).sum()
            }
            
        return outliers

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive statistics."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        statistics = {
            'basic_stats': df[numeric_columns].describe().to_dict(),
            'skewness': df[numeric_columns].skew().to_dict(),
            'kurtosis': df[numeric_columns].kurtosis().to_dict(),
            'unique_counts': df.nunique().to_dict()
        }
        
        # Add time-based statistics
        if isinstance(df.index, pd.DatetimeIndex):
            statistics['temporal'] = {
                'start_date': df.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': df.index.max().strftime('%Y-%m-%d %H:%M:%S'),
                'time_span': str(df.index.max() - df.index.min()),
                'frequency': pd.infer_freq(df.index)
            }
            
        return statistics

    def _analyze_correlations(self, df: pd.DataFrame) -> List[str]:
        """Analyze and validate feature correlations."""
        warnings = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_columns].corr()
        
        # Check for high correlations
        threshold = self.config['data_quality']['correlation_threshold']
        high_correlations = np.where(np.abs(correlation_matrix) > threshold)
        
        for i, j in zip(*high_correlations):
            if i != j:  # Exclude self-correlations
                warnings.append(
                    f"High correlation ({correlation_matrix.iloc[i, j]:.2f}) detected between "
                    f"{correlation_matrix.index[i]} and {correlation_matrix.columns[j]}"
                )
                
        return warnings

    def _validate_ranges(self, df: pd.DataFrame) -> List[str]:
        """Validate value ranges for known variables."""
        errors = []
        range_validations = {
            'TEMPERATURA DO AR - BULBO SECO HORARIA °C': (-40, 50),
            'UMIDADE RELATIVA DO AR HORARIA %': (0, 100),
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': (800, 1100),
            'PRECIPITACÃO TOTAL HORÁRIO MM': (0, 500)
        }
        
        for column, (min_val, max_val) in range_validations.items():
            if column in df.columns:
                invalid_values = df[(df[column] < min_val) | (df[column] > max_val)][column]
                if not invalid_values.empty:
                    errors.append(
                        f"Invalid values in {column}: {len(invalid_values)} values outside "
                        f"range [{min_val}, {max_val}]"
                    )
                    
        return errors

    def _calculate_quality_score(self, df: pd.DataFrame, missing_data: Dict, outliers: Dict) -> float:
        """Calculate overall data quality score."""
        weights = {
            'missing_data': 0.4,
            'outliers': 0.3,
            'time_continuity': 0.3
        }
        
        # Missing data score
        missing_score = 1 - missing_data['total_missing_rate']
        
        # Outlier score
        total_outliers = sum(sum(method.values()) for method in outliers.values())
        outlier_rate = total_outliers / (len(df) * len(outliers))
        outlier_score = 1 - outlier_rate
        
        # Time continuity score
        if isinstance(df.index, pd.DatetimeIndex):
            expected_intervals = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
            continuity_score = len(df) / len(expected_intervals)
        else:
            continuity_score = 1.0
            
        # Calculate weighted score
        quality_score = (
            weights['missing_data'] * missing_score +
            weights['outliers'] * outlier_score +
            weights['time_continuity'] * continuity_score
        )
        
        return quality_score

    def _log_validation_results(self, report: ValidationReport):
        """Log validation results."""
        self.logger.info("=== Data Validation Report ===")
        self.logger.info(f"Validation Status: {'PASSED' if report.is_valid else 'FAILED'}")
        self.logger.info(f"Data Quality Score: {report.data_quality_score:.2f}")
        
        if report.errors:
            self.logger.error("Validation Errors:")
            for error in report.errors:
                self.logger.error(f"  - {error}")
                
        if report.warnings:
            self.logger.warning("Validation Warnings:")
            for warning in report.warnings:
                self.logger.warning(f"  - {warning}")
                
        self.logger.info("Missing Data Summary:")
        self.logger.info(f"  Total Missing Rate: {report.missing_data['total_missing_rate']:.2%}")
        
        self.logger.info("Outlier Summary:")
        for column, stats in report.outliers.items():
            self.logger.info(f"  {column}: {stats['zscore_outliers']} z-score outliers")

if __name__ == "__main__":
    # Example usage
    config_manager = ConfigManager()
    logger = logging.getLogger(__name__)
    
    validator = DataValidator(config_manager, logger)
    
    # Load the actual data
    data_path = "./Scripts/all_data/csvs_concatenated/concatenated_full/full_concatenated.csv.gz"
    df = pd.read_csv(data_path, compression='gzip')
    
    validation_report = validator.validate_dataset(df)