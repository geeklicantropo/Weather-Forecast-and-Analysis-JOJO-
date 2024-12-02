import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("concatenation.log"), logging.StreamHandler()]
)

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
treated_folder = os.path.join(base_dir, 'csvs_treated')

def create_concatenation_folders():
    """Create folders for different types of concatenation."""
    concat_folders = {
        '5year': os.path.join(treated_folder, 'concat_5years'),
        '10year': os.path.join(treated_folder, 'concat_10years'),
        'full': os.path.join(treated_folder, 'concat_full')
    }
    
    for folder in concat_folders.values():
        os.makedirs(folder, exist_ok=True)
        
    return concat_folders

def extract_year_from_filename(filename: str) -> int:
    """Extract year from filename."""
    # Pattern for 'INMET_CO_DF_A001_BRASILIA_01-01-2001_A_31-12-2001.CSV'
    match = re.search(r'_\d{2}-\d{2}-(\d{4})\.CSV$', filename.upper())
    if match:
        return int(match.group(1))
    
    logging.error(f"Could not extract year from filename: {filename}")
    raise ValueError(f"Invalid filename format: {filename}")

def group_files_by_year(files: list) -> dict:
    """Group files by their year."""
    year_groups = {}
    for file in files:
        try:
            year = extract_year_from_filename(file)
            year_groups.setdefault(year, []).append(file)
        except ValueError as e:
            logging.error(f"Error grouping file {file}: {str(e)}")
            continue
    return year_groups

def group_by_period(year_groups: dict, period: int) -> dict:
    """Group files by period (5 or 10 years)."""
    period_groups = {}
    for year, files in year_groups.items():
        period_start = (year // period) * period
        period_end = period_start + (period - 1)
        key = (period_start, period_end)
        if key not in period_groups:
            period_groups[key] = []
        period_groups[key].extend(files)
    return period_groups

def concatenate_files(files: list, output_path: str, source_folder: str, pbar: tqdm) -> bool:
    """Concatenate a group of files."""
    try:
        # Process first file to get headers
        first_file = files[0]
        first_file_path = os.path.join(source_folder, first_file)
        
        # Read and write first file
        df = pd.read_csv(first_file_path)
        df.to_csv(output_path, index=False)
        del df
        
        # Update progress
        file_size = os.path.getsize(first_file_path)
        pbar.update(file_size)
        
        # Process remaining files
        for file in files[1:]:
            file_path = os.path.join(source_folder, file)
            
            # Read and append in chunks
            for chunk in pd.read_csv(file_path, chunksize=10000):
                chunk.to_csv(output_path, mode='a', header=False, index=False)
            
            # Update progress
            file_size = os.path.getsize(file_path)
            pbar.update(file_size)
            
        return True
        
    except Exception as e:
        logging.error(f"Error in concatenation: {str(e)}")
        return False

def perform_period_concatenation(period: int, source_folder: str, output_folder: str, 
                               year_groups: dict) -> bool:
    """Perform concatenation for a specific period."""
    try:
        period_groups = group_by_period(year_groups, period)
        
        # Calculate total size
        total_size = sum(
            os.path.getsize(os.path.join(source_folder, f))
            for files in period_groups.values()
            for f in files
        )
        
        with tqdm(total=total_size, 
                 unit='B', 
                 unit_scale=True, 
                 desc=f"Concatenating {period}-year periods") as pbar:
            
            for (start_year, end_year), files in period_groups.items():
                output_file = os.path.join(
                    output_folder,
                    f'concatenated_{start_year}_to_{end_year}.csv'
                )
                
                if os.path.exists(output_file):
                    # Update progress for skipped files
                    skipped_size = sum(
                        os.path.getsize(os.path.join(source_folder, f)) 
                        for f in files
                    )
                    pbar.update(skipped_size)
                    logging.info(f"Skipping existing file: {output_file}")
                    continue
                
                logging.info(f"Processing period {start_year}-{end_year}")
                if not concatenate_files(files, output_file, source_folder, pbar):
                    return False
                
        return True
        
    except Exception as e:
        logging.error(f"Error in period concatenation: {str(e)}")
        return False

def perform_full_concatenation(source_folder: str, output_folder: str, 
                             all_files: list) -> bool:
    """Perform full concatenation of all files."""
    output_file = os.path.join(output_folder, 'full_concatenated.csv')
    
    if os.path.exists(output_file):
        logging.info("Full concatenated file already exists. Skipping.")
        return True
    
    total_size = sum(os.path.getsize(os.path.join(source_folder, f)) 
                     for f in all_files)
    
    with tqdm(total=total_size, 
             unit='B', 
             unit_scale=True, 
             desc="Creating full concatenation") as pbar:
        
        return concatenate_files(all_files, output_file, source_folder, pbar)

def main():
    """Main function to coordinate concatenations."""
    # Create concatenation folders
    concat_folders = create_concatenation_folders()
    
    # Get list of treated CSV files
    csv_files = [f for f in os.listdir(treated_folder) 
                 if f.lower().endswith('.csv') and 
                 os.path.isfile(os.path.join(treated_folder, f))]
    
    if not csv_files:
        logging.error("No treated CSV files found to concatenate.")
        return
    
    # Group files by year
    year_groups = group_files_by_year(csv_files)
    
    # Perform 5-year concatenation
    logging.info("Starting 5-year concatenation...")
    if not perform_period_concatenation(5, treated_folder, concat_folders['5year'], 
                                      year_groups):
        return
    
    # Perform 10-year concatenation
    logging.info("Starting 10-year concatenation...")
    if not perform_period_concatenation(10, treated_folder, concat_folders['10year'], 
                                      year_groups):
        return
    
    # Perform full concatenation
    logging.info("Starting full concatenation...")
    if not perform_full_concatenation(treated_folder, concat_folders['full'], 
                                    csv_files):
        return
    
    logging.info("All concatenations completed successfully")

if __name__ == "__main__":
    main()