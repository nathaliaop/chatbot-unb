import pandas as pd

# Load the dataset
df = pd.read_csv('metrics.csv')

df = df.drop(columns=['Answer relevancy RAG Gemma Distiluse.1','Faithfulness RAG Gemma Distiluse.1','Contextual precision RAG Gemma Distiluse.1','Contextual recall RAG Gemma Distiluse.1','Contextual relevancy RAG Gemma Distiluse.1'])

# Save to new dataset
df.to_csv('metrics.csv', index=False)