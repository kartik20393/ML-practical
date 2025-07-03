import pandas as pd
csv_data = pd.read_csv('train.csv')
excel_data = pd.read_excel('train.xlsx',sheet_name='train')
for data, name in [(csv_data,'CSV'), (excel_data,'Excel')]:
    print(f"First few rows of {name} file:")
    print(data.head())
    print(f"\nSummary statistics of {name} file:")
    print(data.describe())
    print(f"\nInformation about columns in {name} file:")
    print(data.info())
