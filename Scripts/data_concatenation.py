import os
import pandas as pd
from tqdm import tqdm
import logging
import re
import shutil  # Para copiar e comprimir arquivos existentes
import gzip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
base_dir = './Scripts/all_data'  # Ajuste o diretório base conforme necessário
treated_folder = os.path.join(base_dir, 'csvs_treated')       # Pasta de origem
concat_folder = os.path.join(base_dir, 'csvs_concatenated')   # Pasta de destino

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
    # 'DATA DE FUNDACAO': 'datetime64[ns]',  # Removido, não é necessário
    # 'YEAR': 'Int32'  # Será tratado dinamicamente
}

def create_concatenation_folders():
    """Create concatenation subfolders."""
    os.makedirs(concat_folder, exist_ok=True)
    concat_folders = {
        '5year': os.path.join(concat_folder, 'concatenated_5years'),
        '10year': os.path.join(concat_folder, 'concatenated_10years'),
        'full': os.path.join(concat_folder, 'concatenated_full'),
        'testa_modelo': os.path.join(concat_folder, 'test_models')
    }
    for folder in concat_folders.values():
        os.makedirs(folder, exist_ok=True)
    return concat_folders

def extract_year_from_filename(filename: str) -> int:
    """Extract the year from the filename."""
    # Remove a extensão do arquivo
    if filename.endswith('.csv.gz'):
        filename_no_ext = filename[:-7]
    elif filename.endswith('.csv'):
        filename_no_ext = filename[:-4]
    else:
        logging.error(f"Could not extract year from filename: {filename}")
        raise ValueError(f"Invalid filename format for extracting year: {filename}")
    
    # Extrai os 4 últimos caracteres do nome do arquivo sem extensão
    year_str = filename_no_ext[-4:]
    if year_str.isdigit():
        return int(year_str)
    else:
        # Se os 4 últimos caracteres não forem dígitos, tenta extrair o ano do padrão '_YYYY_'
        match = re.search(r'_(\d{4})_', filename_no_ext)
        if match:
            return int(match.group(1))
        else:
            logging.error(f"Could not extract year from filename: {filename}")
            raise ValueError(f"Invalid filename format for extracting year: {filename}")

def add_missing_columns(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Add 'YEAR' column if missing and drop 'DATA DE FUNDACAO' if present."""
    # Drop 'DATA DE FUNDACAO' column if it exists
    if 'DATA DE FUNDACAO' in df.columns:
        df = df.drop(columns=['DATA DE FUNDACAO'])

    # Add 'YEAR' column if missing
    if 'YEAR' not in df.columns:
        try:
            year = extract_year_from_filename(filename)
            df['YEAR'] = year
        except ValueError as e:
            logging.error(f"Error adding 'YEAR' column to file {filename}: {e}")
            raise e
    else:
        # Ensure 'YEAR' is of integer type
        df['YEAR'] = df['YEAR'].astype('Int32')

    return df

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

def group_by_period(year_groups: dict, period: int, min_year: int, max_year: int) -> dict:
    """Group files by specified period (5 or 10 years), adjusting end year if it exceeds max_year."""
    period_groups = {}
    # Ajusta o ano inicial para ser múltiplo do período
    period_start = (min_year // period) * period
    while period_start <= max_year:
        period_end = period_start + (period - 1)
        if period_end > max_year:
            period_end = max_year
        key = (period_start, period_end)
        period_groups[key] = []
        period_start += period

    # Associa os arquivos aos períodos correspondentes
    for year, files in year_groups.items():
        for (start_year, end_year) in period_groups.keys():
            if start_year <= year <= end_year:
                period_groups[(start_year, end_year)].extend(files)
                break
    return period_groups

def concatenate_files(files: list, output_path: str, compressed: bool = False):
    """Concatenate a list of files into a single CSV."""
    try:
        # Verifica se o arquivo já existe
        if os.path.exists(output_path):
            logging.info(f"File {output_path} already exists. Skipping concatenation.")
            return

        # Initialize progress bar
        total_size = sum(os.path.getsize(os.path.join(treated_folder, f)) for f in files)
        pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Concatenating into {output_path}")

        # Identify datetime columns
        datetime_columns = [col for col, dtype in dtype_mapping.items() if dtype == 'datetime64[ns]']
        non_datetime_dtypes = {col: dtype for col, dtype in dtype_mapping.items() if dtype != 'datetime64[ns]'}

        # Prepare compression settings
        compression = 'gzip' if compressed else None

        first_chunk = True  # To write headers only once

        # Iterate over the files and append to the output file
        for file in files:
            file_path = os.path.join(treated_folder, file)
            file_size = os.path.getsize(file_path)
            # Determina o tipo de compressão com base na extensão do arquivo
            compression_type = 'gzip' if file.endswith('.gz') else None
            for chunk in pd.read_csv(
                file_path,
                chunksize=100000,
                dtype=non_datetime_dtypes,
                compression=compression_type,
                parse_dates=datetime_columns,
            ):
                # Add missing columns and drop unnecessary ones
                chunk = add_missing_columns(chunk, file)
                # Append to the output CSV
                chunk.to_csv(
                    output_path,
                    mode='a',
                    index=False,
                    header=first_chunk,
                    compression=compression,
                    encoding='utf-8'
                )
                first_chunk = False  # Only write header for the first chunk
            # Update progress bar
            pbar.update(file_size)
        pbar.close()
    except Exception as e:
        logging.error(f"Error concatenating files into {output_path}: {e}")
        raise e

def compress_file(input_path: str, output_path: str):
    """Compress an existing CSV file to a .gz file."""
    try:
        if os.path.exists(output_path):
            logging.info(f"Compressed file {output_path} already exists. Skipping compression.")
            return
        logging.info(f"Compressing {input_path} into {output_path}")
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logging.info(f"File compressed successfully into {output_path}")
    except Exception as e:
        logging.error(f"Error compressing file {input_path} into {output_path}: {e}")
        raise e

def create_testa_modelo(output_folder: str):
    """Create the testa_modelo file with 100 random samples per year."""
    try:
        # Path to the full_concatenated.csv.gz file
        full_concatenated_path = os.path.join(concat_folders['full'], 'full_concatenated.csv.gz')

        if not os.path.exists(full_concatenated_path):
            # Verifica se o arquivo não comprimido existe
            uncompressed_full_path = os.path.join(concat_folders['full'], 'full_concatenated.csv')
            if os.path.exists(uncompressed_full_path):
                # Comprime o arquivo existente
                compress_file(uncompressed_full_path, full_concatenated_path)
            else:
                logging.error("full_concatenated.csv or full_concatenated.csv.gz not found. Cannot create testa_modelo.")
                return

        logging.info("Creating testa_modelo...")
        
        # Identify datetime columns explicitly
        datetime_columns = [col for col, dtype in dtype_mapping.items() if dtype == 'datetime64[ns]']
        non_datetime_dtypes = {col: dtype for col, dtype in dtype_mapping.items() if dtype != 'datetime64[ns]'}
        
        chunks = pd.read_csv(
            full_concatenated_path,
            chunksize=100000,
            dtype=non_datetime_dtypes,
            parse_dates=datetime_columns,
            compression='gzip'
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
        testa_modelo_file = os.path.join(output_folder, 'testa_modelo.csv.gz')

        # Save the file in gzip-compressed format
        logging.info(f"Saving testa_modelo to {testa_modelo_file}...")
        sampled_data.to_csv(testa_modelo_file, index=False, compression='gzip', encoding='utf-8')
        logging.info("testa_modelo created successfully.")

    except Exception as e:
        logging.error(f"Error creating testa_modelo: {e}")
        raise e

def perform_period_concatenation(period: int, output_folder: str, year_groups: dict):
    """Perform concatenation for a specific period."""
    years_in_data = sorted(year_groups.keys())
    min_year_in_data = min(years_in_data)
    max_year_in_data = max(years_in_data)
    period_groups = group_by_period(year_groups, period, min_year_in_data, max_year_in_data)
    for (start_year, end_year), files in period_groups.items():
        if files:  # Verifica se há arquivos para concatenar nesse período
            output_file = os.path.join(output_folder, f'concatenated_{start_year}_to_{end_year}.csv.gz')
            # Verifica se o arquivo já existe
            if os.path.exists(output_file):
                logging.info(f"File {output_file} already exists. Skipping concatenation.")
                continue
            logging.info(f"Concatenating files from {start_year} to {end_year} into {output_file}")
            concatenate_files(files, output_file, compressed=True)

def perform_full_concatenation(all_files: list, output_folder: str):
    """Perform full concatenation of all files."""
    output_file = os.path.join(output_folder, 'full_concatenated.csv.gz')
    # Verifica se o arquivo já existe
    if os.path.exists(output_file):
        logging.info(f"File {output_file} already exists. Skipping concatenation.")
        return
    # Verifica se o arquivo não comprimido existe e precisa ser comprimido
    uncompressed_output_file = os.path.join(output_folder, 'full_concatenated.csv')
    if os.path.exists(uncompressed_output_file):
        logging.info(f"Uncompressed file {uncompressed_output_file} exists. Compressing it.")
        compress_file(uncompressed_output_file, output_file)
        return
    logging.info(f"Concatenating all files into {output_file}")
    concatenate_files(all_files, output_file, compressed=True)

def main():
    """Main function to coordinate concatenations."""
    global concat_folders
    concat_folders = create_concatenation_folders()
    csv_files = [f for f in os.listdir(treated_folder) if f.lower().endswith('.csv.gz') or f.lower().endswith('.csv')]
    
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
    
    logging.info("All processes completed successfully.")

if __name__ == '__main__':
    main()
