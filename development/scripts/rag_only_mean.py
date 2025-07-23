import pandas as pd

# Load the dataset
df = pd.read_csv('metrics.csv')

# Suponha que `df` tenha colunas: pergunta, bleu, rouge, meteor
# E que o nome do modelo seja, por exemplo, "modelo-x"

model_name = "RAG Mistral MiniLM"
means = df[['Answer relevancy RAG Mistral MiniLM','Faithfulness RAG Mistral MiniLM','Contextual precision RAG Mistral MiniLM', 'Contextual recall RAG Mistral MiniLM', 'Contextual relevancy RAG Mistral MiniLM']].mean()

# Converte para DataFrame com uma linha e adiciona o nome do modelo
# df_means = pd.read_csv('means.csv') # pd.DataFrame([means])
row = {
    'Name':'RAG Mistral MiniLM',
    'Answer relevancy': means['Answer relevancy RAG Mistral MiniLM'],
    'Faithfulness': means['Faithfulness RAG Mistral MiniLM'],
    'Contextual precision': means['Contextual precision RAG Mistral MiniLM'],
    'Contextual recall': means['Contextual recall RAG Mistral MiniLM'],
    'Contextual relevancy': means['Contextual relevancy RAG Mistral MiniLM'],
}
df_means = pd.read_csv('meansllm.csv')
df_means = pd.concat([df_means, pd.DataFrame([row])], ignore_index=True)
# df_means.insert(1, 'Name', model_name)
df_means.to_csv('meansllm.csv', index=False)