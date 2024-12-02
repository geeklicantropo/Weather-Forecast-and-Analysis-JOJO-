import os
import shutil
import zipfile
import pandas as pd
from tqdm import tqdm
import logging
import chardet
import re
import csv
import io

# Configurar o logging - limitado para mostrar apenas mensagens de alto nível
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Caminhos
data_folder = './Scripts/all_data/dados'
temp_folder = './Scripts/all_data/temp_extracted_files'  # Pasta temporária para arquivos extraídos
output_folder = './Scripts/all_data/csvs_processed'      # Pasta para armazenar cada CSV processado
final_output_file = './Scripts/all_data/concatenated_data.csv'

# Garante que a pasta temporária esteja vazia removendo-a, se existir, e recriando-a
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
os.makedirs(temp_folder, exist_ok=True)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Funções auxiliares
def standardize_column_names(columns):
    return [col.upper().replace(",", "").replace("(", "").replace(")", "").replace("Ç", "C").strip() for col in columns]

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def extract_metadata_from_filename(filename):
    pattern = r"INMET_([A-Z]{1,2})_([A-Z]{2})_([A-Z0-9]+)_(.*?)_\d{2}-\d{2}-\d{4}"
    match = re.search(pattern, filename)
    if not match:
        return {}

    metadata_keys = ["REGIAO", "UF", "CODIGO (WMO)", "ESTACAO"]
    metadata_values = match.groups()
    metadata_dict = dict(zip(metadata_keys, metadata_values))
    return metadata_dict

# Mapeamento de tipos
tipos = {
    'REGIAO': 'string',
    'UF': 'string',
    'ESTACAO': 'string',
    'CODIGO (WMO)': 'string',
    'LATITUDE': 'float64',
    'LONGITUDE': 'float64',
    'ALTITUDE': 'float64',  # Alterado para float64
    'DATA DE FUNDACAO': 'Int64'  # Armazena o ano como inteiro, permitindo NaNs
}

# Ordena os arquivos .zip em ordem lexicográfica e processa cada um
for zip_filename in sorted(os.listdir(data_folder)):
    if zip_filename.endswith('.zip'):
        year = os.path.splitext(zip_filename)[0]
        zip_path = os.path.join(data_folder, zip_filename)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extracted_paths = zip_ref.namelist()
            zip_ref.extractall(temp_folder)

            # Processa cada arquivo CSV extraído do zip
            files_to_process = []
            for path in extracted_paths:
                full_path = os.path.join(temp_folder, path)
                if os.path.isfile(full_path) and path.upper().endswith('.CSV'):
                    files_to_process.append(full_path)
                elif os.path.isdir(full_path):
                    # Se o zip contiver um diretório, obtém os CSVs dentro dele
                    for root, dirs, files in os.walk(full_path):
                        for file in files:
                            if file.upper().endswith('.CSV'):
                                files_to_process.append(os.path.join(root, file))

            # Configura a barra de progresso TQDM para o arquivo zip
            with tqdm(total=len(files_to_process), desc=f"Processando {zip_filename}", leave=True) as zip_progress:
                for csv_file in files_to_process:
                    output_csv_path = os.path.join(output_folder, os.path.basename(csv_file))

                    # Verifica se o CSV já foi processado antes de qualquer outra operação
                    if os.path.exists(output_csv_path):
                        zip_progress.update(1)  # Atualiza o progresso mesmo se pular
                        continue  # Pula o processamento deste arquivo se já existir

                    try:
                        # Detecta a codificação e lê os metadados
                        encoding = detect_encoding(csv_file)
                        with open(csv_file, 'r', encoding=encoding, errors='replace') as f:
                            metadata_lines = [next(f).strip() for _ in range(8)]
                            metadata = []
                            for line in metadata_lines:
                                parts = line.split(":", 1)
                                if len(parts) > 1:
                                    value = parts[1].strip().lstrip(';').strip()
                                else:
                                    value = ""
                                metadata.append(value)
                            csv_content = f.read()

                        # Processa os metadados e as linhas de dados
                        metadata_columns = ["REGIAO", "UF", "ESTACAO", "CODIGO (WMO)", "LATITUDE", "LONGITUDE", "ALTITUDE", "DATA DE FUNDACAO"]
                        metadata_dict = dict(zip(metadata_columns, metadata))
                        filename_metadata = extract_metadata_from_filename(os.path.basename(csv_file))

                        # Preenche metadados ausentes a partir do nome do arquivo
                        for key, value in filename_metadata.items():
                            if not metadata_dict.get(key):
                                metadata_dict[key] = value

                        # Processa e converte os valores dos metadados
                        for key in metadata_columns:
                            val = metadata_dict.get(key, "")
                            if val == "":
                                metadata_dict[key] = None
                            elif key in ['LATITUDE', 'LONGITUDE']:
                                val = val.replace(',', '.').lstrip(';').strip()
                                try:
                                    metadata_dict[key] = float(val)
                                except ValueError:
                                    metadata_dict[key] = None
                            elif key == 'ALTITUDE':
                                val = val.replace(',', '.').lstrip(';').strip()
                                try:
                                    metadata_dict[key] = float(val)  # Alterado para float
                                except ValueError:
                                    metadata_dict[key] = None
                            elif key == 'DATA DE FUNDACAO':
                                val = val.lstrip(';').strip()
                                try:
                                    # Analisar a data com um formato especificado
                                    date = pd.to_datetime(val, format='%d/%m/%Y', errors='coerce')
                                    if pd.notnull(date):
                                        metadata_dict[key] = date.year
                                    else:
                                        # Se a análise falhar, extrai o ano do nome do arquivo
                                        file_year_match = re.search(r'\d{4}', os.path.basename(csv_file))
                                        metadata_dict[key] = int(file_year_match.group()) if file_year_match else None
                                except Exception:
                                    metadata_dict[key] = None
                            else:
                                metadata_dict[key] = val

                        # Lê os dados do CSV
                        csv_buffer = io.StringIO(csv_content)
                        reader = csv.reader(csv_buffer, delimiter=';', quotechar='"')
                        rows = list(reader)
                        if not rows:
                            logging.warning(f"Nenhuma linha de dados em {csv_file}")
                            # Mesmo sem dados, cria um DataFrame vazio com as colunas corretas
                            data = pd.DataFrame(columns=standardize_column_names([]))
                        else:
                            header = standardize_column_names(rows[0])
                            expected_num_columns = len(header)
                            cleaned_rows = []

                            for row in rows[1:]:
                                # Garante que a linha tenha o número correto de colunas
                                if len(row) != expected_num_columns:
                                    if len(row) > expected_num_columns:
                                        row = row[:expected_num_columns]
                                    else:
                                        row += [''] * (expected_num_columns - len(row))
                                cleaned_rows.append(row)

                            # Cria o DataFrame
                            data = pd.DataFrame(cleaned_rows, columns=header)

                        # Adiciona as colunas de metadados
                        for col, val in metadata_dict.items():
                            data[col] = val

                        # Remove colunas 'Unnamed'
                        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

                        # Define os tipos de dados
                        for col, dtype in tipos.items():
                            if col in data.columns:
                                data[col] = data[col].astype(dtype)

                        data['YEAR'] = int(year)
                        # Salva no CSV
                        data.to_csv(output_csv_path, index=False)

                        

                        # Limpa a memória para cada CSV processado
                        del data

                        # Atualiza a barra de progresso
                        zip_progress.update(1)

                    except Exception as e:
                        logging.error(f"Erro ao processar {csv_file}: {e}")
                        # Você pode implementar lógica adicional aqui para lidar com o erro

            # Limpa os arquivos temporários para cada zip após o processamento
            shutil.rmtree(temp_folder)
            os.makedirs(temp_folder, exist_ok=True)

# Limpeza final
shutil.rmtree(temp_folder)
