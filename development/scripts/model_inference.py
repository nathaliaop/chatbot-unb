from transformers import AutoTokenizer, AutoModelForCausalLM
from numba import cuda
from huggingface_hub import login
from dotenv import load_dotenv
import pandas as pd
import os
import torch
import gc

# Load environment varaibles
load_dotenv()

# Free space in CUDA
device = cuda.get_current_device()
device.reset()

# Login to Hugging Face
login(token=os.environ['HUGGING_FACE_TOKEN'])

# Load the dataset
df = pd.read_csv('key.csv')
docu = df.to_dict('records')

# Choose the fine-tuned model
name = "nathaliaop/chatbot_unb_llama8B"
# name = "nathaliaop/chatbot_unb_mistral"
# name = "nathaliaop/chatbot_unb_gemma"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(name).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(name)

for index, row in df.iterrows():
    torch.cuda.empty_cache()
    
    # Optional: run garbage collector
    gc.collect()

    print(index)
    question = row['Pergunta']

    # str = "Quais são as áreas de pesquisa do professor Vinícius Ruela Pereira Borges?"
    # str = "Quais são os cursos oferecidos pela UnB?"
    # str = "Quais são os horarios de funcionamento da UnB?"
    # str = "Onde a professora Alba Cristina fez graduacao?"
    # str = "As disciplinas cursadas no exterior podem ser aproveitadas?"
    # str = "O que signifaca a sigla CiC?"
    # str = "O que signifaca a sigla FT?"

    # strs = [
    #     "As disciplinas cursadas no exterior podem ser aproveitadas?",
    #     "O que signifaca a sigla CiC?",
    #     "O que signifaca a sigla FT?",
    # ]

    # str = "Quais artigos o professor Vinícius Ruela já escreveu"
    # str = "O que e mobilidade academica"
    # str = "Qual o horario de funcionamento da unb"
    # str = "Porque não tem vaga para todos os cursos no processo de mudança de curso?"

    # Configure the answer generation
    messages = [
        {"role": "user", "content": question},
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

    with torch.no_grad():
        # Generate answer
        generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens = 4096, pad_token_id = tokenizer.eos_token_id)
        response_ids = generated_ids[:, input_ids.shape[-1]:]
        answer = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]

        df.at[index, 'FT Llama'] = answer
        # Save to new dataset
        df.to_csv('key.csv', index=False)

        # Clear memory-intensive variables
        del messages, input_ids, generated_ids, response_ids, answer
        
        # Empty the CUDA cache to free up unused memory
        torch.cuda.empty_cache()
        
        # Optional: run garbage collector
        gc.collect()