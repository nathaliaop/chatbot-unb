from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from numba import cuda
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
from dotenv import load_dotenv
import pandas as pd
import os
import gc
import torch

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Load environment varaibles
load_dotenv()

def createPrompt(question, context):
  return f'\n\nContexto:\n\n{context}\n\nPergunta: {question}'

# Free space in CUDA
device = cuda.get_current_device()
device.reset()
torch.cuda.empty_cache()
gc.collect()

# Login to Hugging Face
login(token=os.environ['HUGGING_FACE_TOKEN'])

# Choose the encoder
# encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device='cpu')
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device='cpu')
# encoder = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2", device='cpu')

# Load the dataset
df = pd.read_csv('key2.csv')
docu = df.to_dict('records')

# Connect to RAG database
qclient = QdrantClient(
    url=os.environ['QDRANT_DATABASE_URL'],
    api_key=os.environ['QDRANT_API_KEY'],
    port=None,
    timeout=100,
)

# Create a collection based on the loaded dataset
collection_name="chatbot-unb"

if qclient.collection_exists(collection_name=collection_name):
    qclient.delete_collection(collection_name=collection_name)

if not qclient.collection_exists(collection_name=collection_name):
    qclient.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
            distance=models.Distance.COSINE,
        ),
    )

qclient.upload_points(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=idx, vector=encoder.encode(str(doc['Pergunta'])).tolist(), payload=doc
        )
        for idx, doc in enumerate(docu)
    ],
)

# Choose a model to generate the answer using the documents retrieved with RAG
# name = "mistralai/Mistral-7B-Instruct-v0.3"
# name = "google/gemma-7b-it"
# name = "meta-llama/Llama-2-7b-chat-hf"
name = "unsloth/mistral-7b-instruct-v0.3"
# name = "unsloth/llama-2-7b-chat"
# name = "unsloth/gemma-7b-it"

# Unsloth can load these models in 4bit for efficiency
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = name,
    max_seq_length = 4096,
    dtype = torch.float16,
    load_in_4bit = True,
)

# Load model and tokenizer
# model = AutoModelForCausalLM.from_pretrained(name)#.to("cuda")
# tokenizer = AutoTokenizer.from_pretrained(name)

# Enable native 2x faster inference
FastLanguageModel.for_inference(model)

for index, row in df.loc[689:].iterrows(): # df.loc[1488:1803].iterrows():
    # Empty the CUDA cache to free up unused memory
    torch.cuda.empty_cache()
    
    # Optional: run garbage collector
    gc.collect()

    print(index)
    # print(row['Pergunta'], row['Resposta'])

    # question = "Qual é o horário de funcionamento da UnB?"
    # question = "Quais artigos o professor Vinícius Ruela Pereira Borges escreveu?"
    # question = "Qual é a área de pesquisa do professor Vinícius Ruela Pereira Borges?"
    
    # Retrieve documents with RAG
    question = row['Pergunta']
    hits = qclient.query_points(
        collection_name=collection_name,
        query=encoder.encode(question).tolist(),
        limit=3,
    )

    context = ''
    for i in range(len(hits.points)):
        context += hits.points[i].payload['Pergunta'] + hits.points[i].payload['Resposta']
        df.at[index, f'Context RAG Gemma {i} MPNet'] = hits.points[i].payload['Resposta']

    context = context[:4096]

    # Configure the answer generation
    messages = [
        {"role": "user", "content": createPrompt(question, context)},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize = False
    )

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    # Generate answer
    with torch.no_grad():
        generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens = 4096, pad_token_id = tokenizer.eos_token_id)
        response_ids = generated_ids[:, input_ids.shape[-1]:]
        answer = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]

        df.at[index, 'RAG Gemma MPNet'] = answer
        # Save to new dataset
        df.to_csv('key2.csv', index=False)

        # Clear memory-intensive variables
        del hits, messages, input_ids, generated_ids, response_ids, answer
        
        # Empty the CUDA cache to free up unused memory
        torch.cuda.empty_cache()
        
        # Optional: run garbage collector
        gc.collect()