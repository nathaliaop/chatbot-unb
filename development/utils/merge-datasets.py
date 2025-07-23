import pandas as pd

# Load the dataset
df1 = pd.read_csv('metrics.csv')
df2 = pd.read_csv('metrics2.csv')

df3 = pd.concat([df1, df2[['Answer relevancy RAG Llama MiniLM','Faithfulness RAG Llama MiniLM','Contextual precision RAG Llama MiniLM','Contextual recall RAG Llama MiniLM','Contextual relevancy RAG Llama MiniLM','Answer relevancy RAG Gemma MiniLM','Faithfulness RAG Gemma MiniLM','Contextual precision RAG Gemma MiniLM','Contextual recall RAG Gemma MiniLM','Contextual relevancy RAG Gemma MiniLM']]], axis=1)

# Save to new dataset
df3.to_csv('metrics.csv', index=False)