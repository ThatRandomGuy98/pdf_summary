# Thesis RAG Assistant (Early Demo)

This is a **very early proof-of-concept** of a local **Retrieval-Augmented Generation (RAG)** setup that allows asking questions about a PDF document (a master’s thesis) using **LangChain**, **Ollama**, and **Chroma**.

The goal is purely exploratory: load a thesis, index it locally, and query it via a simple command-line interface.

---

## What this does

- Loads a PDF thesis
- Splits it into text chunks
- Generates embeddings locally with Ollama
- Stores embeddings in a Chroma vector database
- Retrieves relevant chunks per question
- Uses a local LLM to answer questions based only on retrieved context

This is **not production-ready** and intentionally minimal.

---

## Requirements

- Python 3.10+
- Ollama installed and running

Required Ollama models:
ollama pull llama3.1
ollama pull nomic-embed-text
