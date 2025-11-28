# The system — Retrieval-Augmented Generation System Overview

Document type: Technical Report

Subject: System architecture, ingestion, retrieval, UI

Client: NTT DATA VDS — internal RAG app (not a project)

Date: 28 November 2025

Version: 0.01

Authors:

- Nguyễn Hòa
- Lê Chiến

Contact: [email or phone if needed]

## Table of Contents

| Section | Title                    |
|---------|-------------------------|
| 1       | Introduction            |
| 1.1     | Scope                   |
| 1.2     | Main Features           |
| 2       | How to Use The system      |
| 3       | Project Architecture    |
| 4       | Figures and Tables      |
| 5       | Storage & Configuration |
| 6       | Testing & QA Dashboard  |
| 7       | Appendix                |

## 1. Introduction

The system is a computer system that helps people find answers in PDF documents. It takes many PDF files and turns them into a big, searchable library. When you ask a question, The system looks for the best information by checking both the meaning and the important words in the documents. It then shows you the most useful answers, even if you are not an expert in computers. This tool is made for everyone at NTT DATA VDS to make searching documents easy and fast.

## 1.1 Scope

The system is designed to search and find answers across many PDF documents. The system can read all types of PDFs, including scanned files and tables, ensuring that no important information is missed. Every page, table, and image is processed to collect all available content.

After processing, The system divides the text into small, clear segments. For searching, the system uses two methods: finding answers by the meaning of the question and by matching important words. This approach helps deliver the most relevant results, even when different words are used.

Each answer includes information about its source, such as the document name and page number, allowing results to be verified easily.

The system can also clarify questions to improve answer quality. The system works with advanced language models to generate simple, understandable responses. A dashboard is provided for testing and comparing results.

No technical expertise is required to operate The system. All steps are automatic, and answers are presented directly.

## 1.2 Main Features

**PDFLoader:** The system reads every PDF and collects all important information, including text, tables, images, and where each piece comes from. This helps keep track of the source for every answer and makes searching more reliable.

| Module      | What it does                                                      |
|------------|-------------------------------------------------------------------|
| PDFLoader  | Reads all PDFs, extracts text, tables, images, and source details |
| Chunkers   | Splits text into small, clear pieces for better searching         |
| Embedders  | Changes text into numbers for meaning-based search                |
| BM25       | Finds answers by matching important words                         |
| Hybrid     | Combines meaning-based and keyword search for best results        |
| Query Enh. | Improves questions for better answers                             |
| Fusion     | Merges results from different searches                            |
| Reranking  | Puts the best answers at the top                                  |
| Context    | Prepares a summary for the language model                         |
| LLM        | Writes answers in simple language, shows sources                  |
| Prompt     | Uses special instructions to guide the language model in answering|

Chunking: The system breaks long text into small, easy-to-read pieces. This helps the system find and show only the most useful parts when searching.

Embeddings: The system changes each piece of text into numbers. This lets the computer compare the meaning of different pieces and find answers that match the question, even if the words are different.

Vector search (FAISS): The system uses a special method to quickly find pieces of text that have a similar meaning to the question. This makes searching faster and more accurate.

Keyword search (BM25/Whoosh): The system also looks for important words in the question and matches them with words in the documents. This helps find answers that use the same words typed.

Hybrid search: The system combines both meaning-based search and keyword search. By using both methods, it gives better results than using just one.

Query enhancement: Sometimes, the system can rewrite or expand the question to make searching easier and more effective. This means more helpful answers, even if the question is short or unclear.

Embedding fusion: If there are many ways to ask the same question, the system can combine them into one search. This helps find the best answers, no matter how the question is asked.

Score fusion: The system compares results from different searches and puts them together. This way, the most relevant answers appear at the top.

Reranking: After finding possible answers, the system checks them again and puts the best ones first. This makes sure the most useful information is shown quickly.

Prompt: The system uses special instructions, called prompts, to guide the language model in answering questions. These prompts help the model understand what kind of answer is needed, keep responses clear and relevant, and ensure that important details are included. Prompts can be changed or improved to fit different tasks or user needs.

LLM generation: The system uses smart language models (like Gemini, LMStudio, or Ollama) to write answers in natural, easy-to-understand language. It also shows where the information came from, so results can be trusted.

## 2. How to Use The system

Install and Setup: Create a virtual environment by running `python -m venv .venv`, then activate it on Windows with `.venv\Scripts\Activate.ps1` or on Linux/Mac with `source .venv/bin/activate`; install the required Python packages using `pip install -r requirements.txt`; optionally set up Ollama or other LLM providers if needed; and download language models if prompted.

Configuration (.streamlit/secrets.toml): For full functionality, set up API keys for services like HuggingFace and Google Gemini; copy the secrets template using `cp .streamlit/secrets.example .streamlit/secret.toml`; edit the secrets file with your actual API keys, for example `HF_TOKEN = "your_huggingface_token"` and `gemini_api_key = "your_gemini_key"`; alternatively, set environment variables with `export GOOGLE_API_KEY="your_gemini_key"` and `export HF_TOKEN="your_huggingface_token"`; and never commit actual API keys to version control, as the secrets file is already in .gitignore.

Launch the Application: Start the web interface by running `streamlit run ui/app.py`, and if you want the dashboard for evaluation, launch it with `streamlit run ui/dashboard/app.py`.

Run Embedding: In the web interface (UI), go to the embedding section and click the button to run embedding; this step only needs to be done once for each new set of documents.

Ask Questions and Get Answers: Enter your question in the chat box; the system will search all documents and show the best answers; each answer will include the source document and page number.

Choose Model and Settings: Users can select which language model to use (Gemini, LMStudio, Ollama, etc.) in the settings sidebar, and other options like reranking and query enhancement can also be adjusted.

No extra steps are needed: once installed and embedded, the system is ready for use — just open the app, type a question, and get answers.

## 3 Project Architecture

The project's architecture is modular and organized around clear responsibilities: ingestion (PDF loading and chunking), embedding and vector storage, hybrid retrieval (FAISS + BM25), query enhancement, fusion and reranking, and the UI/LLM layer. These components work together to ensure documents are processed reliably, indexed for fast search, and returned with clear source information so users can trust and verify answers.

### 3.1 Ingestion

The ingestion pipeline, implemented in `PDFLoaders` and the `chunkers` module, reads PDF files, extracts text, tables, and images, and splits content into manageable chunks while recording provenance metadata. This stage includes optional OCR for scanned documents, special handling for tables and figures, and writes processed chunks and metadata to the `data/chunks/` and `data/metadata/` folders so downstream embedding and indexing steps can operate on consistent inputs.

### 3.2 Indexing

The indexing stage converts text chunks into embeddings using configured embedders and stores the resulting vectors in FAISS under `data/vectors/`. At the same time, a lexical index (BM25/Whoosh) is built under `data/bm25_index/` for fast keyword searches. Indexing is designed to be repeatable: if embedders or embedding dimensions change, the FAISS and BM25 indexes can be rebuilt to keep retrieval accurate.


