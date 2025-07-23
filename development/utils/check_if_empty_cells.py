import pandas as pd

df = pd.read_csv('key.csv')

columns = df.columns.tolist()

for index, row in df.iterrows():
    for column in columns:
        # if pd.isna(row[column]) or str(row[column]).strip() == '':
        if (row[column] == None or row[column] == ''):
            print(f'Index {index} column {column} is empty or null')

if (df.isnull().values.any() or (df == "").values.any()):
    print("tem")

# print(df[df.isnull().any(axis=1) | (df == "").any(axis=1)])

empty_cells = []

for idx, row in df.iterrows():
    for col in df.columns:
        if pd.isna(row[col]) or row[col] == "":
            empty_cells.append((idx, col))

print(empty_cells)