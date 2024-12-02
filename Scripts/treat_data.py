import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import re
import warnings

# Suppress warnings (except for errors)
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()]
)

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
processed_folder = os.path.join(base_dir, 'all_data/csvs_processed')
optimized_folder = os.path.join(base_dir, 'all_data/csvs_treated')

# Ensure the optimized folder exists
os.makedirs(optimized_folder, exist_ok=True)

# Define standardized expected columns with appropriate data types
expected_columns = {
    "DATA YYYY-MM-DD": "datetime64[ns]",
    "HORA UTC": "Int32",
    "PRECIPITAÇÃO TOTAL HORÁRIO MM": "float32",
    "PRESSÃO ATMOSFÉRICA AO NÍVEL DA ESTAÇÃO HORÁRIA MB": "float32",
    "PRESSÃO ATMOSFÉRICA MAX. NA HORA ANT. AUT MB": "float32",
    "PRESSÃO ATMOSFÉRICA MIN. NA HORA ANT. AUT MB": "float32",
    "RADIAÇÃO GLOBAL KJ/M²": "float32",
    "TEMPERATURA DO AR - BULBO SECO HORÁRIA °C": "float32",
    "TEMPERATURA DO PONTO DE ORVALHO °C": "float32",
    "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) °C": "float32",
    "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) °C": "float32",
    "UMIDADE RELATIVA DO AR - HORÁRIA %": "Int32",
    "VENTO DIREÇÃO HORÁRIA (GR) ° (GR)": "Int32",
    "VENTO VELOCIDADE HORÁRIA (M/S)": "float32",
    "REGIÃO": "string",
    "UF": "string",
    "ESTAÇÃO": "string",
    "CÓDIGO (WMO)": "string",
    "DATA DE FUNDAÇÃO": "datetime64[ns]",
    "YEAR": "Int32"
}

# Standardize column names function
def standardize_column_name(column):
    return re.sub(r'\s+', ' ', column.strip().upper())

standardized_expected_columns = {standardize_column_name(k): v for k, v in expected_columns.items()}

# Function to parse file name and extract metadata
def parse_filename(filename):
    # Remove file extension
    filename = os.path.splitext(filename)[0]
    # Remove 'INMET_' prefix if present
    if filename.startswith('INMET_'):
        filename = filename[len('INMET_'):]
    # Split the filename into parts
    parts = filename.split('_')
    # REGIÃO, UF, CÓDIGO (WMO), ESTAÇÃO
    if len(parts) >= 6:
        regiao = parts[0]
        uf = parts[1]
        codigo_wmo = parts[2]
        # ESTAÇÃO may contain underscores that are part of the name
        estacao_parts = parts[3:-2]  # Assuming dates are always the last two parts
        estacao = ' '.join(estacao_parts).replace('_', ' ').strip()
    else:
        regiao = uf = codigo_wmo = estacao = None
    return regiao, uf, codigo_wmo, estacao

# Process each CSV file in the processed folder
processed_files = sorted([f for f in os.listdir(processed_folder) if f.lower().endswith('.csv')])

if not processed_files:
    logging.warning(f"No CSV files found in {processed_folder}. Please check the folder and try again.")
else:
    with tqdm(total=len(processed_files), desc="Optimizing CSV files") as pbar:
        for csv_file in processed_files:
            input_path = os.path.join(processed_folder, csv_file)
            output_path = os.path.join(optimized_folder, csv_file)

            if os.path.exists(output_path):
                pbar.update(1)
                continue

            try:
                # Parse the filename to get metadata
                regiao, uf, codigo_wmo, estacao = parse_filename(csv_file)

                # Load CSV with all columns as strings initially
                df = pd.read_csv(input_path, dtype=str)
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns

                # Standardize column names
                df.columns = [standardize_column_name(col) for col in df.columns]

                # Identify missing columns
                missing_columns = [col for col in standardized_expected_columns if col not in df.columns]

                # Add missing columns with placeholders
                for col in missing_columns:
                    df[col] = pd.NA  # Use pandas NA value

                # Fill REGIÃO, UF, CÓDIGO (WMO), ESTAÇÃO if they are null or missing
                for col, value in zip(
                    ["REGIÃO", "UF", "CÓDIGO (WMO)", "ESTAÇÃO"],
                    [regiao, uf, codigo_wmo, estacao]
                ):
                    if col in df.columns:
                        df[col] = df[col].fillna(value)
                    else:
                        df[col] = value

                # Ensure all expected columns are in the DataFrame
                df = df[list(standardized_expected_columns.keys())]

                # Convert columns to specified types
                for col, col_type in standardized_expected_columns.items():
                    if col in df.columns:
                        if "float" in col_type or "Int" in col_type:
                            # Clean the data by removing unwanted characters
                            df[col] = df[col].str.replace(r'[^\d\.\-]', '', regex=True)
                            # Remove empty strings
                            df[col] = df[col].replace('', pd.NA)
                            # Attempt to convert to numeric type with errors='raise'
                            try:
                                df[col] = pd.to_numeric(
                                    df[col],
                                    errors='raise',
                                    downcast="float" if "float" in col_type else "integer"
                                )
                            except Exception as e:
                                # Find the problematic entries
                                invalid_entries = df[col][~df[col].str.match(r'^-?\d+(\.\d+)?$', na=False)]
                                logging.error(f"Invalid numeric data in column '{col}' of file '{csv_file}':\n{invalid_entries}")
                                raise
                        elif "datetime" in col_type:
                            # Attempt to convert to datetime with errors='raise'
                            try:
                                df[col] = pd.to_datetime(df[col], errors='raise')
                            except Exception as e:
                                logging.error(f"Invalid datetime data in column '{col}' of file '{csv_file}':\n{df[col]}")
                                raise
                        else:
                            df[col] = df[col].astype("string")

                # Suppress pandas SettingWithCopyWarning
                pd.options.mode.chained_assignment = None

                # Save the optimized CSV
                df.to_csv(output_path, index=False)
                pbar.update(1)
                del df

            except Exception as e:
                logging.error(f"Error processing {csv_file}: {e}")
                raise  # Re-raise the exception to prevent skipping errors

    logging.info(f"Optimization complete. All optimized files saved to the '{optimized_folder}' folder.")
