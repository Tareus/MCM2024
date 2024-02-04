import pandas as pd

csv_file = '..\\data\\R7M1.csv'
data = pd.read_csv(csv_file)
xls_file = '..\\data\\R7M1.xlsx'
data.to_excel(xls_file, index=False)
