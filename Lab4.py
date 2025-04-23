import pandas as pd 
# Load CSV file 
csv_data = pd.read_csv('train.csv')   
# Display the first few rows of the CSV file 
print("First few rows of CSV file:") 
print(csv_data.head()) 
# Summary statistics 
print("\nSummary statistics of CSV file:") 
print(csv_data.describe()) 
# Information about columns 
print("\nInformation about columns in CSV file:") 
print(csv_data.info())
# Load Excel file 
excel_data = pd.read_excel('train.xlsx',sheet_name='train')   
# Display the first few rows of the Excel file 
print("First few rows of Excel file:") 
print(excel_data.head()) 
# Summary statistics 
print("\nSummary statistics of Excel file:") 
print(excel_data.describe()) 
# Information about columns 
print("\nInformation about columns in Excel file:") 
print(excel_data.info())
