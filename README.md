# AskMyPDF AI using LangChain

AskMyPDF AI is a mini Generative AI project where users can upload a PDF and ask questions from its content.

## Features

* Upload PDF
* Ask questions from PDF
* Get AI-generated answers
* Simple Streamlit interface
* Uses RAG for document-based answering

## Technologies Used

* Python
* Streamlit
* LangChain
* Gemini API
* FAISS
* PyPDFLoader
* Google Generative AI Embeddings

## How It Works

1. User uploads a PDF.
2. PDF text is extracted using PyPDFLoader.
3. Text is split into small chunks.
4. Embeddings are created.
5. FAISS stores the PDF data.
6. User asks a question.
7. LangChain retrieves relevant content.
8. Gemini generates the final answer.

## Installation

```bash
pip install -r requirements.txt
```

## Add API Key

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

Do not upload `.env` on GitHub.

## Run Project

```bash
python -m streamlit run app.py
```

## Learning

Through this project, I learned how LangChain, RAG, embeddings, FAISS, and Gemini API work together to build a PDF question-answering chatbot.

## Author

Sani Devi
