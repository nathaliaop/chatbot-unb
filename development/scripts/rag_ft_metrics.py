from huggingface_hub import login
from dotenv import load_dotenv
import pandas as pd
import evaluate
import os

# Use old version of deepeval with this script

# Load environment varaibles
load_dotenv()

# Login to Hugging Face
login(token=os.environ['HUGGING_FACE_TOKEN'])

# Load the dataset
df = pd.read_csv('key.csv')
metrics = pd.read_csv('metrics.csv')

# Load the BLEU and ROUGE metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
perplexity_metric = evaluate.load("perplexity")

# RAG Llama MiniLM

for index, row in df.iterrows():
    # print(row['Pergunta'], row['Resposta'])
    print(index)

    # Example sentences (non-tokenized)
    reference = [row['Resposta']]
    candidate = [row['RAG Llama MiniLM']]

    # BLEU expects plain text inputs
    bleu_results = bleu_metric.compute(predictions=candidate, references=reference)
    # print(f"BLEU Score: {bleu_results['bleu']:.2f}")
    metrics.at[index, 'BLEU RAG Llama MiniLM'] = bleu_results['bleu']

    # ROUGE expects plain text inputs
    rouge_results = rouge_metric.compute(predictions=candidate, references=reference)
    # Access ROUGE scores (no need for indexing into the result)
    # print(f"ROUGE-1 F1 Score: {rouge_results['rouge1']:.2f}")
    # print(f"ROUGE-L F1 Score: {rouge_results['rougeL']:.2f}")
    metrics.at[index, 'ROUGE RAG Llama MiniLM'] = rouge_results['rougeL']

    meteor_results = meteor_metric.compute(predictions=candidate, references=reference)
    # print(f"METEOR Score: {meteor_results['meteor']:.2f}")
    metrics.at[index, 'METEOR RAG Llama MiniLM'] = meteor_results['meteor']

    # perplexity_results = perplexity_metric.compute(predictions=candidate, model_id='meta-llama/Llama-2-7b-chat-hf')
    # print(f"Perpleexity Score: {perplexity_results['perplexity']:.2f}")

    # Save to new dataset
    metrics.to_csv('metrics.csv', index=False)
