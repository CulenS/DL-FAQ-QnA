# **DL FAQ Answering System with Fine-Tuned FLAN-T5 and Cosine Similarity Evaluation**

This project implements a **Question-Answering System** using a **fine-tuned FLAN-T5** model for generating answers to user queries. The system uses **Cosine Similarity** as a metric to evaluate the relevance of generated responses. It is deployed as a **FastAPI** application for real-time querying.

### **Technologies Used**
- **FLAN-T5** (Fine-tuned version for conditional text generation)
- **Sentence-Transformers** (For generating embeddings)
- **Cosine Similarity** (To evaluate answer relevance)
- **FastAPI** (For creating the API endpoint)
- **Transformers** (For pre-trained models like T5)
- **Pandas & JSON** (For data handling)
- **Python** (For backend logic)

### **Project Overview**
This system takes a user query, processes it, generates an answer using the FLAN-T5 model, and evaluates the relevance of the answer using Cosine Similarity between the query and generated response. It is ideal for providing automated answers to frequently asked questions (FAQs) or any scenario where you have a large set of predefined answers.

### **Features**
- **Query Processing**: Processes the input query for better handling by the FLAN-T5 model.
- **Answer Generation**: Generates an answer using the fine-tuned FLAN-T5 model.
- **Cosine Similarity Evaluation**: Measures how relevant the generated answer is to the user query based on cosine similarity between their embeddings.
- **FastAPI Backend**: Exposes an API to interact with the system in real-time.

### **Setup and Installation**

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <project_directory>
