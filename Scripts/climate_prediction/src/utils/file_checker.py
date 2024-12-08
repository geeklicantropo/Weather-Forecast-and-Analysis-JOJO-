# Scripts/climate_prediction/src/utils/file_checker.py
import os
from typing import Dict, List, Tuple
from pathlib import Path
import logging

class FileChecker:
    def __init__(self, base_dir: str = "Scripts/climate_prediction/outputs/data"):
        self.base_dir = Path(base_dir)
        self.stage_dependencies = {
            'split': [],
            'processed': ['split'],
            'preprocessed': ['processed'],
            'final': ['preprocessed']
        }
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
        if stage not in self.required_files or dataset not in self.required_files[stage]:
            raise ValueError(f"Invalid stage '{stage}' or dataset '{dataset}'")
        return self.base_dir / self.required_files[stage][dataset]
    
    def check_stage_files(self, stage: str, dataset_type: str = None) -> Dict[str, bool]:
        """Check existence of files for a specific stage and optionally specific dataset type."""
        results = {}
        for dataset, filename in self.required_files[stage].items():
            if dataset_type and dataset != dataset_type:
                continue
            file_path = self.base_dir / filename
            results[dataset] = file_path.exists()
            if file_path.exists():
                # Check if file is not empty
                if file_path.stat().st_size == 0:
                    results[dataset] = False
        return results
    
    def check_stage_dependencies(self, target_stage: str, dataset_type: str = None) -> Tuple[bool, str]:
        """Check if all dependencies for a stage are met."""
        if target_stage not in self.stage_dependencies:
            return False, f"Invalid stage: {target_stage}"
            
        for dep_stage in self.stage_dependencies[target_stage]:
            stage_files = self.check_stage_files(dep_stage, dataset_type)
            if not all(stage_files.values()):
                missing = [f for f, exists in stage_files.items() if not exists]
                return False, f"Missing {dep_stage} files: {', '.join(missing)}"
                
        return True, "All dependencies met"
    
    def should_process_stage(self, stage: str, dataset_type: str = None) -> Tuple[bool, str]:
        """Determine if a stage needs processing and why."""
        # Check if target files already exist
        existing_files = self.check_stage_files(stage, dataset_type)
        if all(existing_files.values()):
            return False, f"Files for stage '{stage}' already exist and are valid"
            
        # Check dependencies
        deps_ok, deps_message = self.check_stage_dependencies(stage, dataset_type)
        if not deps_ok:
            return False, deps_message
            
        return True, "Processing needed"
    
    def get_last_completed_stage(self, dataset_type: str = None) -> str:
        """Determine the last completely processed stage."""
        stage_order = ['final', 'preprocessed', 'processed', 'split']
        
        for stage in stage_order:
            if all(self.check_stage_files(stage, dataset_type).values()):
                return stage
        
        return None
    
    def validate_file_size_reduction(self, prev_stage: str, curr_stage: str, 
                                   dataset_type: str) -> bool:
        """Verify file size reduction between stages (specifically for aggregation)."""
        prev_file = self.get_file_path(prev_stage, dataset_type)
        curr_file = self.get_file_path(curr_stage, dataset_type)
        
        if not (prev_file.exists() and curr_file.exists()):
            return False
            
        prev_size = prev_file.stat().st_size
        curr_size = curr_file.stat().st_size
        
        # For preprocessed stage, expect significant reduction due to aggregation
        if curr_stage == 'preprocessed':
            return curr_size < (prev_size * 0.9)  # Expect at least 10% reduction
            
        return True
    
    def check_final_exists(self) -> bool:
        """Check if final processed files exist and are valid."""
        final_files = self.check_stage_files('final')
        if not all(final_files.values()):
            return False
            
        # Verify file sizes
        for dataset, exists in final_files.items():
            if exists:
                file_path = self.get_file_path('final', dataset)
                if file_path.stat().st_size == 0:
                    return False
                    
        return True