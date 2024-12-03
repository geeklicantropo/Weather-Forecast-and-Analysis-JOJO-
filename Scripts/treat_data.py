import os
import pandas as pd
from tqdm import tqdm
import csv

# Paths
base_dir = './Scripts/all_data'
processed_folder = os.path.join(base_dir, 'csvs_processed')  # Source folder
treated_folder = os.path.join(base_dir, 'csvs_treated')     # Destination folder

# Ensure the 'csvs_treated' folder exists
os.makedirs(treated_folder, exist_ok=True)

# Data type mapping
dtype_mapping = {
    'DATA': 'datetime64[ns]',
    'HORA': 'string',
    'PRECIPITACAO TOTAL HORARIO MM': 'float32',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': 'float32',
    'PRESSAO ATMOSFERICA MAX. NA HORA ANT. AUT MB': 'float32',
    'PRESSAO ATMOSFERICA MIN. NA HORA ANT. AUT MB': 'float32',
    'RADIACAO GLOBAL KJ/M²': 'float32',
    'TEMPERATURA DO AR - BULBO SECO HORARIA °C': 'float32',
    'TEMPERATURA DO PONTO DE ORVALHO °C': 'float32',
    'TEMPERATURA MAXIMA NA HORA ANT. (AUT) °C': 'float32',
    'TEMPERATURA MINIMA NA HORA ANT. (AUT) °C': 'float32',
    'TEMPERATURA ORVALHO MAX. NA HORA ANT. AUT °C': 'float32',
    'TEMPERATURA ORVALHO MIN. NA HORA ANT. AUT °C': 'float32',
    'UMIDADE REL. MAX. NA HORA ANT. AUT %': 'float32',
    'UMIDADE REL. MIN. NA HORA ANT. AUT %': 'float32',
    'UMIDADE RELATIVA DO AR - HORARIA %': 'Int32',
    'VENTO DIRECAO HORARIA (GR) ° (GR)': 'Int32',
    'VENTO RAJADA MAXIMA M/S': 'float32',
    'VENTO VELOCIDADE HORARIA (M/S)': 'float32',
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

def detect_delimiter(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = f.read(2048)  # Read a small sample of the file
        sniffer = csv.Sniffer()
        try:
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter
        except csv.Error:
            return ','  # Default to comma if detection fails
        
def process_csv_files():
    """Process each CSV file to reduce size and save them in 'csvs_treated'."""
    csv_files = [f for f in os.listdir(processed_folder) if f.lower().endswith('.csv')]

    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            input_path = os.path.join(processed_folder, csv_file)
            output_filename = os.path.splitext(csv_file)[0] + '.csv.gz'
            output_path = os.path.join(treated_folder, output_filename)

            # Detect delimiter
            delimiter = detect_delimiter(input_path)
            df = pd.read_csv(input_path, sep=delimiter, dtype=str, encoding='utf-8')

            # Normalize column names
            df.columns = df.columns.str.upper()

            # Remove "Unnamed" columns
            unnamed_cols = [col for col in df.columns if col.startswith("UNNAMED")]
            df.drop(columns=unnamed_cols, inplace=True)

            # Replace commas with dots in all data (string level)
            df = df.replace(',', '.', regex=True)

            # Process and convert data types
            for col in df.columns:
                col_upper = col.upper()
                if col_upper in dtype_mapping:
                    target_type = dtype_mapping[col_upper]
                    if target_type == 'float32':
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
                    elif target_type == 'Int32':
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int32')
                    elif target_type == 'datetime64[ns]':
                        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                    elif target_type == 'string':
                        df[col] = df[col].astype('string')

            # Enforce downcast to reduce precision for all numeric types
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')

            # Verify column types
            #print(f"Final types for file {csv_file}:")
            #print(df.dtypes)

            # Write the DataFrame to a compressed CSV file
            df.to_csv(
                output_path,
                index=False,         # Do not write row indices
                sep=',',             # Use commas as delimiters
                quoting=csv.QUOTE_MINIMAL,  # Quote only fields with special characters
                compression='gzip',  # Compress the output
                encoding='utf-8'     # UTF-8 encoding
            )

        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")

if __name__ == '__main__':
    process_csv_files()
