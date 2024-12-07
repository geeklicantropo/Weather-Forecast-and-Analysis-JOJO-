import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
from datetime import datetime, timedelta
import gc
import psutil
from tqdm import tqdm

@dataclass
class ValidationReport:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict
    missing_data: Dict
    outliers: Dict
    data_quality_score: float
    validation_stages: List[str]
    memory_usage: Dict

class DataValidator:
    def __init__(self, config_manager, logger, chunk_size: int = 20000):
        self.config = config_manager.get_validation_config()
        self.preprocessing_config = config_manager.get_preprocessing_config()
        self.logger = logger
        self.chunk_size = chunk_size
        self.target_variable = self.preprocessing_config['target_variable']
        self.required_columns = self._get_required_columns()
        self.validation_stages = []
        self.validation_stats = {
            'processed_chunks': 0,
            'total_rows': 0,
            'current_memory': 0,
            'peak_memory': 0,
            'error_chunks': 0
        }

    def _get_required_columns(self) -> set:
        """Define required columns based on configuration."""
        return {
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

    def _update_statistics(self, current_stats: Dict, new_stats: Dict) -> None:
        """Update running statistics with new chunk statistics."""
        if not current_stats:
            current_stats.update(new_stats)
            return

        # Update basic stats
        for column, stats in new_stats['basic_stats'].items():
            if column not in current_stats['basic_stats']:
                current_stats['basic_stats'][column] = stats
            else:
                # Update mean, std, min, max
                n1 = current_stats['basic_stats'][column]['count']
                n2 = stats['count']
                total_n = n1 + n2
                
                # Update mean
                m1 = current_stats['basic_stats'][column]['mean']
                m2 = stats['mean']
                current_stats['basic_stats'][column]['mean'] = (n1*m1 + n2*m2) / total_n
                
                # Update min/max
                current_stats['basic_stats'][column]['min'] = min(
                    current_stats['basic_stats'][column]['min'], 
                    stats['min']
                )
                current_stats['basic_stats'][column]['max'] = max(
                    current_stats['basic_stats'][column]['max'], 
                    stats['max']
                )
                current_stats['basic_stats'][column]['count'] = total_n

        # Update skewness and kurtosis
        for metric in ['skewness', 'kurtosis']:
            if metric not in current_stats:
                current_stats[metric] = new_stats[metric]
            else:
                for column, value in new_stats[metric].items():
                    if column not in current_stats[metric]:
                        current_stats[metric][column] = value
                    else:
                        current_stats[metric][column] = (
                            current_stats[metric][column] + value
                        ) / 2

        # Update unique counts
        for column, count in new_stats['unique_counts'].items():
            if column not in current_stats['unique_counts']:
                current_stats['unique_counts'][column] = count
            else:
                current_stats['unique_counts'][column] = max(
                    current_stats['unique_counts'][column],
                    count
                )

    def _update_outliers(self, current_outliers: Dict, new_outliers: Dict) -> None:
        """Update running outlier counts with new chunk outliers."""
        if not current_outliers:
            current_outliers.update(new_outliers)
            return

        for column, outlier_types in new_outliers.items():
            if column not in current_outliers:
                current_outliers[column] = outlier_types
            else:
                for outlier_type, count in outlier_types.items():
                    if outlier_type not in current_outliers[column]:
                        current_outliers[column][outlier_type] = count
                    else:
                        current_outliers[column][outlier_type] += count

    def _calculate_quality_score(self, statistics: Dict, missing_data: Dict, 
                               outliers: Dict) -> float:
        """Calculate overall data quality score."""
        weights = {
            'missing_data': 0.4,
            'outliers': 0.3,
            'statistics': 0.3
        }
        
        # Missing data score
        missing_score = 1 - missing_data['total_missing_rate']
        
        # Outlier score
        total_outliers = sum(
            sum(method.values()) 
            for outlier_dict in outliers.values() 
            for method in [outlier_dict]
        )
        total_records = statistics['basic_stats'][self.target_variable]['count']
        outlier_score = 1 - (total_outliers / (total_records * len(outliers)))
        
        # Statistics score based on completeness and validity
        stats_scores = []
        for column in self.required_columns:
            if column in statistics['basic_stats']:
                completeness = 1 - (
                    statistics['basic_stats'][column].get('null_count', 0) / 
                    total_records
                )
                validity = 1 - (
                    outliers.get(column, {}).get('iqr_outliers', 0) / 
                    total_records
                )
                stats_scores.append((completeness + validity) / 2)
        
        statistics_score = np.mean(stats_scores) if stats_scores else 0.0
        
        # Calculate weighted final score
        quality_score = (
            weights['missing_data'] * missing_score +
            weights['outliers'] * outlier_score +
            weights['statistics'] * statistics_score
        )
        
        return quality_score
    
    def _initialize_validation_stats(self, file_path: str) -> None:
        """Initialize validation statistics."""
        self.validation_stats['total_rows'] = sum(1 for _ in pd.read_csv(file_path, chunksize=self.chunk_size))
        self.validation_stats['processed_chunks'] = 0
        self.validation_stats['error_chunks'] = 0
        self.validation_stats['start_time'] = datetime.now()

    def _initialize_validation_data(self) -> Dict:
        """Initialize containers for validation data."""
        return {
            'errors': [],
            'warnings': [],
            'statistics': {},
            'missing_data': {'total_missing_rate': 0},
            'outliers': {},
            'validation_stages': set()
        }

    def _update_validation_data(self, validation_data: Dict, chunk_report: ValidationReport) -> None:
        """Update validation data with chunk results."""
        validation_data['errors'].extend(chunk_report.errors)
        validation_data['warnings'].extend(chunk_report.warnings)
        self._update_statistics(validation_data['statistics'], chunk_report.statistics)
        self._update_outliers(validation_data['outliers'], chunk_report.outliers)
        validation_data['missing_data']['total_missing_rate'] += chunk_report.missing_data['total_missing_rate']
        validation_data['validation_stages'].update(chunk_report.validation_stages)

    def _update_validation_stats(self, chunk: pd.DataFrame) -> None:
        """Update validation statistics."""
        self.validation_stats['processed_chunks'] += 1
        self.validation_stats['current_memory'] = psutil.Process().memory_info().rss / 1024 / 1024
        self.validation_stats['peak_memory'] = max(
            self.validation_stats['peak_memory'],
            self.validation_stats['current_memory']
        )

    def _handle_chunk_error(self, chunk_idx: int, error: Exception) -> None:
        """Handle chunk processing errors."""
        self.validation_stats['error_chunks'] += 1
        self.logger.error(f"Error processing chunk {chunk_idx}: {str(error)}")

    def _create_final_report(self, validation_data: Dict) -> ValidationReport:
        """Create final validation report."""
        if self.validation_stats['processed_chunks'] > 0:
            validation_data['missing_data']['total_missing_rate'] /= self.validation_stats['processed_chunks']
            quality_score = self._calculate_quality_score(
                validation_data['statistics'],
                validation_data['missing_data'],
                validation_data['outliers']
            )
        else:
            quality_score = 0.0

        return ValidationReport(
            is_valid=len(validation_data['errors']) == 0,
            errors=validation_data['errors'],
            warnings=validation_data['warnings'],
            statistics=validation_data['statistics'],
            missing_data=validation_data['missing_data'],
            outliers=validation_data['outliers'],
            data_quality_score=quality_score,
            validation_stages=list(validation_data['validation_stages']),
            memory_usage={
                'current': self.validation_stats['current_memory'],
                'peak': self.validation_stats['peak_memory']
            }
        )

    def validate_file(self, file_path: str) -> ValidationReport:
        """Validate entire file in chunks with improved tracking."""
        try:
            self._initialize_validation_stats(file_path)
            validation_data = self._initialize_validation_data()

            with tqdm(total=self.validation_stats['total_rows'], desc="Validating data") as pbar:
                for i, chunk in enumerate(pd.read_csv(file_path, chunksize=self.chunk_size)):
                    try:
                        chunk_report = self._validate_chunk(chunk)
                        self._update_validation_data(validation_data, chunk_report)
                        self._update_validation_stats(chunk)
                        pbar.update(len(chunk))
                        self._cleanup_memory()
                    except Exception as e:
                        self._handle_chunk_error(i, e)

            return self._create_final_report(validation_data)

        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise

    def _validate_chunk(self, chunk: pd.DataFrame) -> ValidationReport:
        """Validate a single chunk of data."""
        errors = []
        warnings = []
        
        # Schema validation
        schema_errors = self._validate_schema(chunk)
        errors.extend(schema_errors)
        
        # Data type validation
        dtype_errors = self._validate_datatypes(chunk)
        errors.extend(dtype_errors)
        
        # Time continuity validation
        continuity_errors, continuity_warnings = self._validate_time_continuity(chunk)
        errors.extend(continuity_errors)
        warnings.extend(continuity_warnings)
        
        # Missing data analysis
        missing_stats = self._analyze_missing_data(chunk)
        
        # Outlier detection
        outliers = self._detect_outliers(chunk)
        
        # Calculate statistics
        statistics = self._calculate_statistics(chunk)
        
        # Calculate quality score
        quality_score = self._calculate_chunk_quality_score(
            chunk, missing_stats, outliers
        )
        
        return ValidationReport(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            statistics=statistics,
            missing_data=missing_stats,
            outliers=outliers,
            data_quality_score=quality_score,
            validation_stages=['schema', 'dtype', 'continuity', 'missing', 'outliers'],
            memory_usage={'current': psutil.Process().memory_info().rss / 1024 / 1024}
        )

    def _validate_schema(self, df: pd.DataFrame) -> List[str]:
        """Validate dataset schema."""
        errors = []
        missing_columns = self.required_columns - set(df.columns)
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            
        if len(df) == 0:
            errors.append("Empty dataset")
            
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            errors.append(f"Duplicate columns found: {duplicate_columns}")
            
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
        
        if 'DATETIME' in df.columns:
            df = df.sort_values('DATETIME')
            time_diff = pd.to_datetime(df['DATETIME']).diff()
            expected_diff = pd.Timedelta(hours=1)
            
            gaps = time_diff[time_diff > expected_diff]
            if not gaps.empty:
                warnings.extend([
                    f"Time gap of {gap} detected at {timestamp}"
                    for timestamp, gap in gaps.items()
                ])
                
            duplicates = df['DATETIME'].duplicated()
            if duplicates.any():
                errors.append(f"Found {duplicates.sum()} duplicate timestamps")
                
        return errors, warnings

    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict:
        """Analyze missing data patterns."""
        missing_stats = {
            'total_missing_rate': df.isnull().mean().mean(),
            'missing_by_column': df.isnull().mean().to_dict(),
            'consecutive_missing': self._get_consecutive_missing(df)
        }
        return missing_stats

    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers using multiple methods."""
        outliers = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            valid_data = df[column].dropna()
            if len(valid_data) > 0:
                z_scores = np.abs(stats.zscore(valid_data))
                iqr = stats.iqr(valid_data)
                q1 = valid_data.quantile(0.25)
                q3 = valid_data.quantile(0.75)
                
                outliers[column] = {
                    'zscore_outliers': (z_scores > 3).sum(),
                    'iqr_outliers': ((valid_data < (q1 - 1.5 * iqr)) | 
                                   (valid_data > (q3 + 1.5 * iqr))).sum()
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
        
        return statistics

    def _calculate_chunk_quality_score(self, df: pd.DataFrame, 
                                     missing_stats: Dict, 
                                     outliers: Dict) -> float:
        """Calculate quality score for a chunk."""
        # Missing data score (40%)
        missing_score = 1 - missing_stats['total_missing_rate']
        
        # Outlier score (30%)
        total_outliers = sum(sum(stats.values()) for stats in outliers.values())
        outlier_score = 1 - (total_outliers / (len(df) * len(outliers)))
        
        # Continuity score (30%)
        if 'DATETIME' in df.columns:
            time_diff = pd.to_datetime(df['DATETIME']).diff()
            continuity_score = 1 - (time_diff > pd.Timedelta(hours=1)).mean()
        else:
            continuity_score = 1.0
            
        return 0.4 * missing_score + 0.3 * outlier_score + 0.3 * continuity_score

    def _cleanup_memory(self):
        """Clean up memory after chunk processing."""
        gc.collect()
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_usage > 1024:
            self.logger.warning(f"High memory usage: {memory_usage:.2f} MB")

    def _get_peak_memory(self) -> float:
        """Get peak memory usage."""
        return psutil.Process().memory_info().peak_wset / 1024 / 1024

if __name__ == "__main__":
    import logging
    from ..utils.config_manager import ConfigManager
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    config_manager = ConfigManager()
    
    validator = DataValidator(config_manager, logger)
    
    input_files = [
        "Scripts/climate_prediction/outputs/data/train_final.csv.gz",
        "Scripts/climate_prediction/outputs/data/test_final.csv.gz"
    ]
    
    for file_path in input_files:
        report = validator.validate_file(file_path)
        logger.info(f"\nValidation Report for {file_path}:")
        logger.info(f"Valid: {report.is_valid}")
        logger.info(f"Quality Score: {report.data_quality_score:.2f}")
        if report.errors:
            logger.error("Errors found:")
            for error in report.errors:
                logger.error(f"  - {error}")
        if report.warnings:
            logger.warning("Warnings found:")
            for warning in report.warnings:
                logger.warning(f"  - {warning}")