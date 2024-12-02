import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv("treated_csvs/INMET_CO_DF_A001_BRASILIA_01-01-2013_A_31-12-2013.CSV")

print(df.dtypes)