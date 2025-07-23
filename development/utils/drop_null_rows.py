import pandas as pd

# Load the original metrics.csv (the one with the column used for filtering)
df_main = pd.read_csv('metrics.csv')

# Find indices where the column is NOT null — i.e., rows to KEEP
valid_indices = df_main[df_main["Answer relevancy RAG Gemma Distiluse"].notnull()].index

# Save cleaned version of metrics.csv
df_main_cleaned = df_main.loc[valid_indices]
df_main_cleaned.to_csv('metrics2.csv', index=False)

# Now load the second CSV you want to clean the same way
df_other = pd.read_csv('key.csv')  # replace with your actual filename

# Use the same indices to select the same rows
df_other_cleaned = df_other.loc[valid_indices]

# Save the cleaned second CSV
df_other_cleaned.to_csv('key2.csv', index=False)
