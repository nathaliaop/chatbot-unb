import pandas as pd

# Load the dataset
df = pd.read_csv('metrics.csv')

# Suponha que `df` tenha colunas: pergunta, bleu, rouge, meteor
# E que o nome do modelo seja, por exemplo, "modelo-x"

model_name = "RAG Llama MiniLM"
means = df[['BLEU RAG Llama MiniLM','ROUGE RAG Llama MiniLM','METEOR RAG Llama MiniLM']].mean()

# Converte para DataFrame com uma linha e adiciona o nome do modelo
# df_means = pd.read_csv('means.csv') # pd.DataFrame([means])
row = {
    'BLEU': means['BLEU RAG Llama MiniLM'],
    'ROUGE': means['ROUGE RAG Llama MiniLM'],
    'METEOR': means['METEOR RAG Llama MiniLM'],
    'Name':'RAG Llama MiniLM',
}
df_means = pd.read_csv('means.csv')
df_means = pd.concat([df_means, pd.DataFrame([row])], ignore_index=True)
# df_means.insert(1, 'Name', model_name)
df_means.to_csv('means.csv', index=False)