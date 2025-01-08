from fastapi import FastAPI
from pydantic import BaseModel
from model import initialize_tokenizer_and_model, initialize_embedding_model, fine_tune_model
from data_preprocessing import load_dataset, clean_dataset, preprocess_text
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Initialize components
file_path = '/content/faq_data.json'
data = load_dataset(file_path)
df = clean_dataset(data)

# Preprocess the data
questions, answers = preprocess_data(df)

# Initialize Tokenizer, Model, and Embedding Model
tokenizer, model = initialize_tokenizer_and_model()
embedding_model = initialize_embedding_model()

# Fine-tune the model
tokenized_inputs, tokenized_outputs = tokenize_data(df, tokenizer)
model, tokenizer = fine_tune_model(tokenized_inputs, tokenized_outputs, tokenizer, model)

# Load fine-tuned model for inference
fine_tuned_model = T5ForConditionalGeneration.from_pretrained("./flan_t5_finetuned")
fine_tuned_tokenizer = T5Tokenizer.from_pretrained("./flan_t5_finetuned")
generator = pipeline('text2text-generation', model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

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
