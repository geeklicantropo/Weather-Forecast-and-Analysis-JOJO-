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

# Configure logging - limited to show only high-level messages
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
data_folder = './Scripts/dados'
temp_folder = './Scripts/temp_extracted_files'  # Temporary folder for extracted files
output_folder = './Scripts/processed_csvs'      # Folder to store each processed CSV
final_output_file = './Scripts/concatenated_data.csv'

# Ensure temp folder is empty by forcefully removing it if it exists, then recreate
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
os.makedirs(temp_folder)

# Helper functions
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

# Sort .zip files in lexicographical order and process each
for zip_filename in sorted(os.listdir(data_folder)):
    if zip_filename.endswith('.zip'):
        year = os.path.splitext(zip_filename)[0]
        zip_path = os.path.join(data_folder, zip_filename)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extracted_paths = zip_ref.namelist()
            zip_ref.extractall(temp_folder)

            # Process each CSV file in the extracted zip
            files_to_process = [os.path.join(temp_folder, path) for path in extracted_paths if path.endswith('.CSV')]

            # Set up TQDM progress bar for the zip file
            with tqdm(total=len(files_to_process), desc=f"Processing {zip_filename}", leave=True) as zip_progress:
                for csv_file in files_to_process:
                    output_csv_path = os.path.join(output_folder, os.path.basename(csv_file))

                    # Check if the CSV has already been processed before any other operation
                    if os.path.exists(output_csv_path):
                        zip_progress.update(1)  # Progress even if skipping
                        continue  # Skip processing this file if it already exists

                    try:
                        # Detect encoding and read metadata
                        encoding = detect_encoding(csv_file)
                        with open(csv_file, 'r', encoding=encoding) as f:
                            metadata_lines = [next(f).strip() for _ in range(8)]
                            metadata = [line.split(": ", 1)[1] if ": " in line else "" for line in metadata_lines]
                            csv_content = f.read()

                        # Process metadata and data rows
                        metadata_columns = ["REGIAO", "UF", "ESTACAO", "CODIGO (WMO)", "LATITUDE", "LONGITUDE", "ALTITUDE", "DATA DE FUNDACAO"]
                        metadata_dict = dict(zip(metadata_columns, metadata))
                        filename_metadata = extract_metadata_from_filename(os.path.basename(csv_file))
                        for key, value in filename_metadata.items():
                            if metadata_dict.get(key, "") == "":
                                metadata_dict[key] = value

                        csv_buffer = io.StringIO(csv_content)
                        reader = csv.reader(csv_buffer, delimiter=';', quotechar='"')
                        rows = list(reader)
                        header = standardize_column_names(rows[0])
                        expected_num_columns = len(header)
                        cleaned_rows = []

                        for row in rows[1:]:
                            if len(row) != expected_num_columns:
                                row = row[:expected_num_columns] if len(row) > expected_num_columns else row + [''] * (expected_num_columns - len(row))
                            cleaned_rows.append(row)

                        # Create DataFrame, add metadata, and save
                        data = pd.DataFrame(cleaned_rows, columns=header)
                        for col, val in metadata_dict.items():
                            data[col] = val if val else None

                        data['YEAR'] = year
                        data.to_csv(output_csv_path, index=False)

                        # Clear memory for each processed CSV
                        del data
                        
                        # Update the progress bar
                        zip_progress.update(1)

                    except Exception as e:
                        logging.error(f"Error processing {csv_file}: {e}")

        # Clean up temporary files for each zip after processing
        shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)
'''
# Iterative concatenation of all processed CSVs into a single final file
if not os.path.exists(final_output_file):
    processed_csv_files = [os.path.join(output_folder, f) for f in sorted(os.listdir(output_folder)) if f.endswith('.csv')]
    
    if not processed_csv_files:
        logging.error("No processed CSV files found in the output folder. Ensure processing completed successfully.")
        raise ValueError("No processed CSV files found for concatenation.")
    
    logging.warning(f"Found {len(processed_csv_files)} files to concatenate. Starting iterative concatenation...")
    
    # Iterate through each processed CSV file, appending to the final output file
    for i, file in enumerate(processed_csv_files):
        try:
            # Read each file and append to the final output CSV
            chunk = pd.read_csv(file)
            
            # Write header only for the first file, then append without header
            if i == 0:
                chunk.to_csv(final_output_file, mode='w', index=False)
            else:
                chunk.to_csv(final_output_file, mode='a', index=False, header=False)
            
            # Optional: Clear memory after processing each file
            del chunk
            logging.info(f"Appended data from {file}")

        except Exception as e:
            logging.error(f"Error reading {file}: {e}")

    logging.warning(f"Data concatenation complete. Output saved to {final_output_file}")
'''
# Final clean up
shutil.rmtree(temp_folder)
shutil.rmtree(output_folder)
