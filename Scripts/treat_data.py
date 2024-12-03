import os
import pandas as pd
from tqdm import tqdm
import csv

# Paths
base_dir = './Scripts/all_data'
processed_folder = os.path.join(base_dir, 'csvs_processed')  # Source folder
treated_folder = os.path.join(base_dir, 'csvs_treated')     # Destination folder

#processed_folder = os.path.join(base_dir, 'TESTES/nao_alterado')
#treated_folder = os.path.join(base_dir, 'TESTES/alterado')

# Ensure the destination folder exists
os.makedirs(treated_folder, exist_ok=True)

# Data type mapping
dtype_mapping = {
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
    """Detect the delimiter of a CSV file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        try:
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter
        except csv.Error:
            return ','

def preprocess_and_convert_floats(df):
    """Replace commas with dots in numeric columns."""
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].notna().any():  # Only process columns with non-null values
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
    return df

def process_csv_files():
    """Process each CSV file to reduce size and save them in treated folder."""
    csv_files = [f for f in os.listdir(processed_folder) if f.lower().endswith('.csv')]

    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            input_path = os.path.join(processed_folder, csv_file)
            output_filename = os.path.splitext(csv_file)[0] + '.csv.gz'
            output_path = os.path.join(treated_folder, output_filename)

            # Read CSV with detected delimiter
            delimiter = detect_delimiter(input_path)
            df = pd.read_csv(
                input_path,
                sep=delimiter,
                dtype=str,  # Load everything as string initially
                encoding='utf-8'
            )

            # Normalize column names
            df.columns = df.columns.str.upper()

            # Remove unnamed columns
            unnamed_cols = [col for col in df.columns if 'UNNAMED' in col.upper()]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)

            # Preprocess numeric columns (replace commas with dots)
            df = preprocess_and_convert_floats(df)

            # Process DATA column
            if 'DATA' in df.columns:
                df['DATA'] = pd.to_datetime(df['DATA'], format='%Y/%m/%d', errors='coerce').dt.strftime('%Y-%m-%d')

            # Process HORA UTC column
            if 'HORA UTC' in df.columns:
                df['HORA UTC'] = pd.to_numeric(
                    df['HORA UTC'].astype(str).str.extract(r'(\d{4})')[0].str[:2], 
                    errors='coerce'
                ).astype('Int32')

            # Convert data types according to mapping
            for col in df.columns:
                if col in dtype_mapping:
                    try:
                        if dtype_mapping[col] == 'float32':
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
                        elif dtype_mapping[col] == 'Int32':
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int32')
                        elif dtype_mapping[col] == 'string':
                            df[col] = df[col].astype('string')
                        elif dtype_mapping[col] == 'datetime64[ns]':
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception as e:
                        print(f"Error converting column {col}: {str(e)}")

            # Save processed file
            df.to_csv(
                output_path,
                index=False,
                compression='gzip',
                quoting=csv.QUOTE_MINIMAL,
                encoding='utf-8'
            )

        except Exception as e:
            print(f"Error processing file {csv_file}: {str(e)}")
            raise  # Re-raise the exception to see the full error traceback

if __name__ == '__main__':
    process_csv_files()