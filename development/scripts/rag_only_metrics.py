from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
import tiktoken
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os
import time
import threading
save_lock = threading.Lock()  # to avoid concurrent writes

load_dotenv()
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

import openai
import os

class SimpleOpenAIWrapper:
    def __init__(self, model="gpt-3.5-turbo", max_tokens=600, temperature=1.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate(self, prompt: str) -> str:
        prompt_tokens = len(enc.encode(prompt))
        max_model_tokens = 16384
        # Reserve 50 tokens as safety buffer
        max_completion_tokens = max_model_tokens - prompt_tokens - 50  
        if max_completion_tokens <= 0:
            raise ValueError("Prompt too long for model context length.")
        max_completion_tokens = min(max_completion_tokens, self.max_tokens)  # respect configured max tokens
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=1.0,
            stop=["\n\n"]
        )
        return response.choices[0].message.content

def truncate_context_tokens(text):
    return enc.decode(enc.encode(text or "")[:100])

def truncate_test_case_tokens(test_case, model_name="gpt-3.5-turbo", max_total_tokens=15384, reserve_output_tokens=1000):
    enc = tiktoken.encoding_for_model(model_name)
    count = lambda s: len(enc.encode(s or ""))
    
    # Truncate input, expected, actual if needed
    def truncate_to_limit(text, limit):
        return enc.decode(enc.encode(text or "")[:limit])

    input_tokens = count(test_case.input)
    expected_output_tokens = count(test_case.expected_output)
    actual_output_tokens = count(test_case.actual_output)

    # Make sure each of these is reasonable
    test_case.input = truncate_to_limit(test_case.input, 1000)
    test_case.expected_output = truncate_to_limit(test_case.expected_output, 500)
    test_case.actual_output = truncate_to_limit(test_case.actual_output, 500)

    # Now recalculate after truncation
    input_tokens = count(test_case.input)
    expected_output_tokens = count(test_case.expected_output)
    actual_output_tokens = count(test_case.actual_output)
    available_for_context = max_total_tokens - (input_tokens + expected_output_tokens + reserve_output_tokens)

    new_context = []
    running_total = 0
    for ctx in test_case.retrieval_context:
        ctx_tokens = count(ctx)
        if running_total + ctx_tokens <= available_for_context:
            new_context.append(ctx)
            running_total += ctx_tokens
        else:
            break
    test_case.retrieval_context = new_context
    return test_case

llm = SimpleOpenAIWrapper(model="gpt-3.5-turbo", max_tokens=600)

def evaluate_row(index, row, metric_row, total_count):
    try:
        # Skip if all metrics already exist and are not NaN
        if all(pd.notnull([
            metric_row.get('Answer relevancy RAG Mistral MiniLM'),
            metric_row.get('Faithfulness RAG Mistral MiniLM'),
            metric_row.get('Contextual precision RAG Mistral MiniLM'),
            metric_row.get('Contextual recall RAG Mistral MiniLM'),
            metric_row.get('Contextual relevancy RAG Mistral MiniLM'),
        ])):
            print(f"[{index}] Skipping already-evaluated row.")
            return index, {}  # No update

        print(f"[{index}] Starting row...")

        start_time = time.time()
        test_case = LLMTestCase(
            input=truncate_context_tokens(row["Pergunta"]),
            actual_output=truncate_context_tokens(row["RAG Mistral MiniLM"]),
            expected_output=truncate_context_tokens(row["Resposta"]),
            retrieval_context=[truncate_context_tokens(row["Context RAG Mistral 0 MiniLM"])]
        )
        test_case = truncate_test_case_tokens(test_case, model_name="gpt-3.5-turbo", reserve_output_tokens=500)

        # Evaluate metrics
        answer = AnswerRelevancyMetric(model=llm)
        faith = FaithfulnessMetric(model=llm)
        cprec = ContextualPrecisionMetric(model=llm)
        crec = ContextualRecallMetric(model=llm)
        crel = ContextualRelevancyMetric(model=llm)

        answer.measure(test_case)
        faith.measure(test_case)
        cprec.measure(test_case)
        crec.measure(test_case)
        crel.measure(test_case)

        elapsed = time.time() - start_time
        print(f"[{index}] Done in {elapsed:.2f}s ({index + 1}/{total_count})")

        return index, {
            'Answer relevancy RAG Mistral MiniLM': answer.score,
            'Faithfulness RAG Mistral MiniLM': faith.score,
            'Contextual precision RAG Mistral MiniLM': cprec.score,
            'Contextual recall RAG Mistral MiniLM': crec.score,
            'Contextual relevancy RAG Mistral MiniLM': crel.score,
        }

    except Exception as e:
        print(f"[{index}] ❌ Error: {e}")
        return index, {
            'Answer relevancy RAG Mistral MiniLM': None,
            'Faithfulness RAG Mistral MiniLM': None,
            'Contextual precision RAG Mistral MiniLM': None,
            'Contextual recall RAG Mistral MiniLM': None,
            'Contextual relevancy RAG Mistral MiniLM': None,
        }

# Load the dataset
df = pd.read_csv('key.csv')
total_rows = len(df)
df = df.iloc[688:1200]
metrics_df = pd.read_csv('metrics.csv')

print(f"🔎 Starting evaluation of {total_rows} rows using gpt-3.5-turbo...")

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(evaluate_row, index, row, metrics_df.loc[index], total_rows) for index, row in df.iterrows()]
    for i, future in enumerate(as_completed(futures), 1):
        index, result = future.result()
        if result: 
            for col, val in result.items():
                metrics_df.at[index, col] = val

            with save_lock:
                metrics_df.to_csv("metrics.csv", index=False)
        if i % 50 == 0:
            print(f"✅ Completed {index}/{total_rows} rows...")

print("💾 Saving results to metrics_all.csv...")
metrics_df.to_csv("metrics.csv", index=False)
print("✅ All done.")

# Mistral MiniLM 688