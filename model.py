from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from transformers import Trainer, TrainingArguments

# Load and initialize the tokenizer and model
def initialize_tokenizer_and_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    return tokenizer, model

# Tokenize the data
def tokenize_data(df, tokenizer):
    inputs = df['question'].apply(lambda x: f"Question: {x}")
    outputs = df['answer']
    tokenized_inputs = tokenizer(inputs.tolist(), padding=True, truncation=True, return_tensors="pt")
    tokenized_outputs = tokenizer(outputs.tolist(), padding=True, truncation=True, return_tensors="pt")
    return tokenized_inputs, tokenized_outputs

# Fine-tune the model
def fine_tune_model(tokenized_inputs, tokenized_outputs, tokenizer, model):
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
    return model, tokenizer

# Initialize SentenceTransformer model for embeddings
def initialize_embedding_model():
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return embedding_model
