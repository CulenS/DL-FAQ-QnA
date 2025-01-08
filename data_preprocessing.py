import json
import re
import pandas as pd

# Load dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Clean dataset
def clean_dataset(df):
    df_clean = df[df['answer'] != ""]
    df_clean = df_clean.drop_duplicates(subset=['question', 'answer'])
    df_clean = df_clean.reset_index(drop=True)
    return df_clean

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_data(df):
    questions = df['question'].apply(preprocess_text)
    answers = df['answer'].apply(preprocess_text)
    return questions, answers
