# Multimodal PDF RAG Assistant (Handwritten OCR & Visual Retrieval)

A complete, high-performance, and modular **Retrieval-Augmented Generation (RAG)** web application built with **Streamlit**, **LangChain**, **PyMuPDF**, **FAISS**, **Groq (Instant Free)**, **Google Gemini**, and **Vision AI OCR**.

Supports text PDFs, **handwritten notes**, scanned documents, and **visual page image retrieval** in chat responses.

---

## ✨ Features

* **Multi-LLM Provider Support**:
  * ⚡ **Groq Cloud (Instant Free)** — Blazing fast responses using `llama-3.3-70b-versatile` and `qwen/qwen3.6-27b`.
  * ♊ **Google Gemini (Free)** — Powerful responses using `gemini-1.5-flash` with zero billing needed.
  * 🤖 **OpenAI** — Optional support for GPT models.
* **Handwritten & Scanned PDF OCR**: Automatically renders PDF pages to images and uses Vision AI to transcribe handwritten notes, cursive text, and scanned documents.
* **Visual Context & Diagram Retrieval**: Displays original PDF page images and diagrams alongside text answers and citations.
* **Traceable Page Citations**: Know exactly which PDF and page number was used for every answer.
* **Local FAISS Vector Indexing**: High-performance semantic search using `sentence-transformers/all-MiniLM-L6-v2`.

---

## 🏗️ Architecture

```text
                  PDF DOCUMENT(S)
                         |
                         v
            PyMuPDF Page Image Rendering
                         |
           +-------------+-------------+
           |                           |
   [Embedded Text]            [Handwritten / Scanned]
           |                           |
           v                           v
   Standard Extraction           Vision AI OCR (Groq/Gemini)
           |                           |
           +-------------+-------------+
                         |
                         v
            Text Chunking & Image Metadata
                         |
                         v
                FAISS Vector Store
                         |
                         v
     User Question -> Top-K Similarity Search
                         |
                         v
            Answer + Page Image Diagrams
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)

Create a `.env` file in the root folder with your free API keys:

```env
# Free Groq API Key (https://console.groq.com/keys)
GROQ_API_KEY=gsk_your_groq_key_here

# Free Gemini API Key (https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_key_here
```
*(You can also paste API keys directly into the Streamlit sidebar UI).*

### 3. Run Application

```powershell
streamlit run app.py
```

---

## 🧪 Running Tests

Run the automated test suite with `pytest`:

```powershell
python -m pytest tests/ -v
```

---

## 📂 Project Structure

```text
RAG/
├── app.py                     # Streamlit Web Application Interface
├── requirements.txt           # Dependency requirements
├── README.md                  # Project Documentation
├── .env.example               # Environment Variables Template
├── .gitignore                 # Excluded git files
│
├── data/                      # Local runtime storage
│   ├── uploads/               # Uploaded PDF files
│   ├── extracted_images/      # Rendered page images for OCR & visual context
│   └── vectorstore/           # FAISS index storage
│
├── src/                       # Backend RAG logic
│   ├── pdf_loader.py          # PyMuPDF rendering & Vision OCR
│   ├── text_splitter.py       # Recursive text chunker with image metadata
│   ├── embeddings.py          # SentenceTransformer embeddings
│   ├── vector_store.py        # FAISS index management
│   ├── retriever.py           # Similarity search wrapper
│   ├── llm.py                 # Multi-LLM provider & Vision AI transcription
│   ├── rag_pipeline.py        # End-to-end RAG orchestrator
│   └── utils.py               # Paths & citation formatting helpers
│
└── tests/                     # Unit test suite
```
