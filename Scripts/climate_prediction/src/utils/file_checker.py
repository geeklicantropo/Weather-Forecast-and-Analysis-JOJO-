# Scripts/climate_prediction/src/utils/file_checker.py
import os
from typing import Dict, List, Tuple
from pathlib import Path

class FileChecker:
    """Utility class to check and validate pipeline file dependencies."""
    
    def __init__(self, base_dir: str = "Scripts/climate_prediction/outputs/data"):
        self.base_dir = Path(base_dir)
        self.required_files = {
            'split': {
                'train': 'train_full_concatenated.csv.gz',
                'test': 'test_full_concatenated.csv.gz'
            },
            'processed': {
                'train': 'train_processed.csv.gz',
                'test': 'test_processed.csv.gz'
            },
            'preprocessed': {
                'train': 'train_preprocessed.csv.gz',
                'test': 'test_preprocessed.csv.gz'
            },
            'final': {
                'train': 'train_final.csv.gz',
                'test': 'test_final.csv.gz'
            }
        }
        
    def get_file_path(self, stage: str, dataset: str) -> Path:
        """Get full path for a specific file."""
        return self.base_dir / self.required_files[stage][dataset]
    
    def check_files(self, stage: str) -> Dict[str, bool]:
        """Check existence of files for a specific stage."""
        results = {}
        for dataset, filename in self.required_files[stage].items():
            file_path = self.base_dir / filename
            results[dataset] = file_path.exists()
        return results
    
    def check_dependencies(self, target_stage: str) -> Tuple[bool, List[str]]:
        """Check if all necessary files exist for a given stage."""
        stage_order = ['split', 'processed', 'preprocessed', 'final']
        target_idx = stage_order.index(target_stage)
        
        missing_files = []
        for stage in stage_order[:target_idx + 1]:
            stage_files = self.check_files(stage)
            for dataset, exists in stage_files.items():
                if not exists:
                    file_path = self.get_file_path(stage, dataset)
                    missing_files.append(str(file_path))
        
        return len(missing_files) == 0, missing_files
    
    def check_final_exists(self) -> bool:
        """Check if final files exist, indicating previous processing is complete."""
        final_files = self.check_files('final')
        return all(final_files.values())
    
    def get_last_completed_stage(self) -> str:
        """Determine the last completed processing stage."""
        stage_order = ['split', 'processed', 'preprocessed', 'final']
        
        for stage in reversed(stage_order):
            stage_files = self.check_files(stage)
            if all(stage_files.values()):
                return stage
        
        return None
    
    def should_process_stage(self, stage: str) -> Tuple[bool, str]:
        """Determine if a stage needs processing and why."""
        # Check if final files already exist
        if self.check_final_exists():
            return False, "Final files already exist. Processing not needed."
            
        # Check dependencies
        deps_ok, missing = self.check_dependencies(stage)
        if not deps_ok:
            return True, f"Missing dependency files: {', '.join(missing)}"
            
        # Check if current stage files exist
        stage_files = self.check_files(stage)
        if all(stage_files.values()):
            return False, f"Files for stage '{stage}' already exist."
            
        return True, "Processing needed."