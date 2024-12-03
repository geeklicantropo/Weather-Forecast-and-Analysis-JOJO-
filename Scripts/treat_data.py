import os
import pandas as pd
from tqdm import tqdm
import csv

# Paths
base_dir = './Scripts/all_data'
processed_folder = os.path.join(base_dir, 'csvs_processed')
treated_folder = os.path.join(base_dir, 'csvs_treated')

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

# Ensure the destination folder exists with proper permissions
try:
    os.makedirs(treated_folder, mode=0o777, exist_ok=True)
    #print(f"Pasta de destino criada/verificada com sucesso: {treated_folder}")
    #print(f"Permissões da pasta de destino: {oct(os.stat(treated_folder).st_mode)[-3:]}")
except Exception as e:
    raise Exception(f"Erro ao criar/verificar pasta de destino: {str(e)}")

def preprocess_and_convert_floats(df):
    """Replace commas with dots in numeric columns."""
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].notna().any():  # Only process columns with non-null values
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
    return df

def process_csv_files():
    """Process each CSV file to reduce size and save them in treated folder."""
    csv_files = [f for f in os.listdir(processed_folder) if f.lower().endswith('.csv')]
    print(f"Encontrados {len(csv_files)} arquivos CSV para processar")
    
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            input_path = os.path.join(processed_folder, csv_file)
            output_filename = os.path.splitext(csv_file)[0] + '.csv.gz'
            output_path = os.path.join(treated_folder, output_filename)
            
            #print(f"\nProcessando arquivo: {csv_file}")
            #print(f"Caminho de entrada: {input_path}")
            #print(f"Caminho de saída: {output_path}")
            
            # Verificar arquivo de entrada
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")
            
            # Ler CSV
            df = pd.read_csv(input_path, sep=',', dtype=str, encoding='utf-8')
            #print(f"Arquivo lido com sucesso. Shape inicial: {df.shape}")
            
            # Processar DataFrame
            df.columns = df.columns.str.upper()
            
            # Remover colunas sem nome
            unnamed_cols = [col for col in df.columns if 'UNNAMED' in col.upper()]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
                #print(f"Removidas {len(unnamed_cols)} colunas sem nome")

            # Preprocessar dados numéricos
            df = preprocess_and_convert_floats(df)

            # Processar coluna DATA
            if 'DATA' in df.columns:
                df['DATA'] = pd.to_datetime(df['DATA'], format='%Y/%m/%d', errors='coerce').dt.strftime('%Y-%m-%d')
                #print("Coluna DATA processada")

            # Processar coluna HORA UTC
            if 'HORA UTC' in df.columns:
                df['HORA UTC'] = pd.to_numeric(
                    df['HORA UTC'].astype(str).str.extract(r'(\d{4})')[0].str[:2], 
                    errors='coerce'
                ).astype('Int32')
                #print("Coluna HORA UTC processada")

            # Converter tipos de dados
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
                        print(f"Erro ao converter coluna {col}: {str(e)}")
                        continue

            #print(f"Shape final do DataFrame: {df.shape}")
            
            # Garantir que o diretório de saída existe
            os.makedirs(os.path.dirname(output_path), mode=0o777, exist_ok=True)
            
            # Tentar salvar primeiro em um arquivo temporário
            temp_output = output_path + '.tmp'
            #print(f"Salvando arquivo temporário em: {temp_output}")
            
            df.to_csv(
                temp_output,
                index=False,
                compression='gzip',
                quoting=csv.QUOTE_MINIMAL,
                encoding='utf-8'
            )
            
            # Se o arquivo temporário foi criado com sucesso, renomear para o arquivo final
            if os.path.exists(temp_output):
                os.replace(temp_output, output_path)
                #print(f"Arquivo salvo com sucesso: {output_path}")
            else:
                raise Exception("Falha ao criar arquivo temporário")
            
        except Exception as e:
            print(f"ERRO ao processar arquivo {csv_file}: {str(e)}")
            print("Abortando processamento devido a erro...")
            raise

if __name__ == '__main__':
    process_csv_files()