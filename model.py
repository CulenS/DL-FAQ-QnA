import json
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, pipeline
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

file_path = '/content/faq_data.json'
data = load_dataset(file_path)
df = pd.DataFrame(data)

# Clean dataset
df_clean = df[df['answer'] != ""]
df_clean = df_clean.drop_duplicates(subset=['question', 'answer'])
df_clean = df_clean.reset_index(drop=True)

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

questions = df_clean['question'].apply(preprocess_text)
answers = df_clean['answer'].apply(preprocess_text)

# Tokenize the dataset for fine-tuning
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

def tokenize_data(df):
    inputs = df['question'].apply(lambda x: f"Question: {x}")
    outputs = df['answer']
    tokenized_inputs = tokenizer(inputs.tolist(), padding=True, truncation=True, return_tensors="pt")
    tokenized_outputs = tokenizer(outputs.tolist(), padding=True, truncation=True, return_tensors="pt")
    return tokenized_inputs, tokenized_outputs

tokenized_inputs, tokenized_outputs = tokenize_data(df_clean)

# Fine-tune the FLAN-T5 model
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

training_args = TrainingArguments(
    output_dir="./flan_t5_finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_inputs,
    eval_dataset=tokenized_outputs,
    tokenizer=tokenizer
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("./flan_t5_finetuned")
tokenizer.save_pretrained("./flan_t5_finetuned")

# Load fine-tuned model for inference
fine_tuned_model = T5ForConditionalGeneration.from_pretrained("./flan_t5_finetuned")
fine_tuned_tokenizer = T5Tokenizer.from_pretrained("./flan_t5_finetuned")
generator = pipeline('text2text-generation', model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

# Initialize SentenceTransformer model for embedding generation
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# FastAPI application
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def answer_question(query: Query):
    # Preprocess the question
    user_query = preprocess_text(query.question)

    # Generate query embeddings
    query_embedding = embedding_model.encode([user_query])

    # Generate answer using the fine-tuned FLAN-T5 model
    prompt = f"Question: {user_query}"
    response = generator(prompt, max_length=100, num_return_sequences=1)
    generated_response = response[0]['generated_text']

    # Preprocess and generate embeddings for the generated response
    generated_response_clean = preprocess_text(generated_response)
    generated_response_embedding = embedding_model.encode([generated_response_clean])

    # Compute Cosine Similarity between query and generated response
    cosine_sim = cosine_similarity(query_embedding, generated_response_embedding)[0][0]

    # Return the result including cosine similarity score
    return {
        "query": query.question,
        "generated_response": generated_response,
        "cosine_similarity": cosine_sim
    }
