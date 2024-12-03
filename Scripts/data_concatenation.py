import os
import pandas as pd
from tqdm import tqdm
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
base_dir = './Scripts/all_data'  # Adjust the base directory as needed
treated_folder = os.path.join(base_dir, 'csvs_treated')       # Source folder
concat_folder = os.path.join(base_dir, 'csvs_concatenated')   # Destination folder

# Define data type mapping for loading CSVs
dtype_mapping = {
    'DATA YYYY-MM-DD': 'object',
    'HORA UTC': 'object',
    'PRECIPITACÃO TOTAL HORÁRIO MM': 'float32',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': 'float32',
    'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. AUT MB': 'float32',
    'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. AUT MB': 'float32',
    'RADIACAO GLOBAL KJ/M²': 'float32',
    'TEMPERATURA DO AR - BULBO SECO HORARIA °C': 'float32',
    'TEMPERATURA DO PONTO DE ORVALHO °C': 'float32',
    'TEMPERATURA MÁXIMA NA HORA ANT. AUT °C': 'float32',
    'TEMPERATURA MÍNIMA NA HORA ANT. AUT °C': 'float32',
    'TEMPERATURA ORVALHO MAX. NA HORA ANT. AUT °C': 'float32',
    'TEMPERATURA ORVALHO MIN. NA HORA ANT. AUT °C': 'float32',
    'UMIDADE REL. MAX. NA HORA ANT. AUT %': 'float32',
    'UMIDADE REL. MIN. NA HORA ANT. AUT %': 'float32',
    'UMIDADE RELATIVA DO AR HORARIA %': 'float32',
    'VENTO DIRECÃO HORARIA GR ° GR': 'float32',
    'VENTO RAJADA MAXIMA M/S': 'float32',
    'VENTO VELOCIDADE HORARIA M/S': 'float32',
    'LATITUDE': 'float32',
    'LONGITUDE': 'float32',
    'ALTITUDE': 'float32',
    'REGIAO': 'string',
    'UF': 'string',
    'ESTACAO': 'string',
    'CODIGO (WMO)': 'string',
    'DATA DE FUNDACAO': 'datetime64[ns]',
    'YEAR': 'Int32'
}

def create_concatenation_folders():
    """Create concatenation subfolders."""
    os.makedirs(concat_folder, exist_ok=True)
    concat_folders = {
        '5year': os.path.join(concat_folder, 'concat_5years'),
        '10year': os.path.join(concat_folder, 'concat_10years'),
        'full': os.path.join(concat_folder, 'concat_full'),
        'testa_modelo': os.path.join(concat_folder, 'testa_modelo')
    }
    for folder in concat_folders.values():
        os.makedirs(folder, exist_ok=True)
    return concat_folders

def extract_year_from_filename(filename: str) -> int:
    """Extract the year from the filename or from the 'YEAR' column."""
    match = re.search(r'_(\d{4})\.csv(?:\.gz)?$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        file_path = os.path.join(treated_folder, filename)
        try:
            df = pd.read_csv(file_path, usecols=['YEAR'], nrows=1, dtype=dtype_mapping, compression='gzip')
            return int(df['YEAR'].iloc[0])
        except Exception as e:
            logging.error(f"Could not extract year from file {filename}: {e}")
            raise ValueError(f"Invalid filename format or 'YEAR' column missing in {filename}")

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
    """Group files by specified period (5 or 10 years)."""
    period_groups = {}
    for year, files in year_groups.items():
        period_start         = (year // period) * period
        period_end = period_start + (period - 1)
        key = (period_start, period_end)
        if key not in period_groups:
            period_groups[key] = []
        period_groups[key].extend(files)
    return period_groups

def concatenate_files(files: list, output_path: str, compressed: bool = False):
    """Concatenate a list of files into a single CSV."""
    try:
        # Remove the output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Initialize progress bar
        total_size = sum(os.path.getsize(os.path.join(treated_folder, f)) for f in files)
        pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Concatenating into {output_path}")
        
        # Identify datetime columns
        datetime_columns = [col for col, dtype in dtype_mapping.items() if dtype == 'datetime64[ns]']
        non_datetime_dtypes = {col: dtype for col, dtype in dtype_mapping.items() if dtype != 'datetime64[ns]'}

        # Prepare compression settings
        compression = 'gzip' if compressed else None

        # Iterate over the files and append to the output file
        for file in files:
            file_path = os.path.join(treated_folder, file)
            file_size = os.path.getsize(file_path)
            for chunk in pd.read_csv(
                file_path, 
                chunksize=100000, 
                dtype=non_datetime_dtypes, 
                compression='gzip',  # Se os arquivos estão comprimidos
                parse_dates=datetime_columns,
            ):
                # Append to the output CSV
                if not os.path.exists(output_path):
                    chunk.to_csv(output_path, index=False, compression=compression, encoding='utf-8')
                else:
                    chunk.to_csv(output_path, mode='a', header=False, index=False, compression=compression, encoding='utf-8')
            # Update progress bar
            pbar.update(file_size)
        pbar.close()
    except Exception as e:
        logging.error(f"Error concatenating files into {output_path}: {e}")
        raise e

def create_testa_modelo(output_folder: str):
    """Create the testa_modelo file with 100 random samples per year."""
    try:
        # Path to the full_concatenated.csv file
        full_concatenated_path = os.path.join(concat_folder, 'concat_full/full_concatenated.csv')

        if not os.path.exists(full_concatenated_path):
            logging.error("full_concatenated.csv not found. Cannot create testa_modelo.")
            return

        logging.info("Creating testa_modelo...")
        
        # Identifique as colunas de datas explicitamente
        datetime_columns = [col for col, dtype in dtype_mapping.items() if dtype == 'datetime64[ns]']
        non_datetime_dtypes = {col: dtype for col, dtype in dtype_mapping.items() if dtype != 'datetime64[ns]'}
        
        chunks = pd.read_csv(
            full_concatenated_path, 
            chunksize=100000, 
            dtype=non_datetime_dtypes, 
            parse_dates=datetime_columns,  # Passa explicitamente as colunas de data
            compression=None
        )

        # DataFrame to store the sampled data
        sampled_data = pd.DataFrame()

        # Process each chunk and sample random rows by year
        for chunk in chunks:
            if 'YEAR' not in chunk.columns:
                logging.error("Column 'YEAR' not found in the dataset.")
                return
            
            # Sample 100 random rows per year in the chunk
            grouped = chunk.groupby('YEAR')
            sampled_chunk = grouped.apply(lambda x: x.sample(n=100, random_state=42) if len(x) >= 100 else x)
            sampled_chunk = sampled_chunk.reset_index(drop=True)
            
            # Append to the final DataFrame
            sampled_data = pd.concat([sampled_data, sampled_chunk], ignore_index=True)

        # Path to save the testa_modelo file
        testa_modelo_path = os.path.join(concat_folder, 'testa_modelo')
        testa_modelo_file = os.path.join(testa_modelo_path, 'testa_modelo.csv.gz')

        # Save the file in gzip-compressed format
        logging.info(f"Saving testa_modelo to {testa_modelo_file}...")
        sampled_data.to_csv(testa_modelo_file, index=False, compression='gzip', encoding='utf-8')
        logging.info("testa_modelo created successfully.")

    except Exception as e:
        logging.error(f"Error creating testa_modelo: {e}")
        raise e

def perform_period_concatenation(period: int, output_folder: str, year_groups: dict):
    """Perform concatenation for a specific period."""
    period_groups = group_by_period(year_groups, period)
    for (start_year, end_year), files in period_groups.items():
        output_file = os.path.join(output_folder, f'concatenated_{start_year}_to_{end_year}.csv')
        #logging.info(f"Concatenating files from {start_year} to {end_year} into {output_file}")
        concatenate_files(files, output_file)

def perform_full_concatenation(all_files: list, output_folder: str):
    """Perform full concatenation of all files."""
    output_file = os.path.join(output_folder, 'full_concatenated.csv')
    #logging.info(f"Concatenating all files into {output_file}")
    concatenate_files(all_files, output_file)

def main():
    """Main function to coordinate concatenations."""
    concat_folders = create_concatenation_folders()
    csv_files = [f for f in os.listdir(treated_folder) if f.lower().endswith('.csv.gz')]
    
    if not csv_files:
        logging.error("No treated CSV files found to concatenate.")
        return
    
    year_groups = group_files_by_year(csv_files)
    
    logging.info("Starting 5-year concatenation...")
    perform_period_concatenation(5, concat_folders['5year'], year_groups)
    
    logging.info("Starting 10-year concatenation...")
    perform_period_concatenation(10, concat_folders['10year'], year_groups)
    
    logging.info("Starting full concatenation...")
    perform_full_concatenation(csv_files, concat_folders['full'])
    
    logging.info("Starting creation of testa_modelo...")
    create_testa_modelo(concat_folders['testa_modelo'])
    
    logging.info("All concatenations completed successfully.")

if __name__ == '__main__':
    main()

