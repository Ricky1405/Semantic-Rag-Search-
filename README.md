Local-Embedding RAG System

A high-performance Retrieval-Augmented Generation (RAG) application designed for privacy-centric document intelligence.
This system performs local vector embedding to eliminate API costs during indexing and uses GitHub Models (GPT-4o-mini) for accurate, cost-efficient natural language generation.

✨ Key Features

🔐 100% Local Embeddings – No sensitive data leaves your machine during vectorization

⚡ Low-Latency Retrieval – Optimized chunking and cosine similarity search

💸 Cost-Efficient – Zero embedding API cost, lightweight LLM for generation

🧠 Hallucination Control – Confidence gating with strict prompt grounding

♻️ Incremental Indexing – Atomic updates with deterministic chunk hashing

🏗️ System Architecture

The application follows a modular, scalable pipeline optimized for performance and reliability.

1. Data Ingestion Layer

Text Loader

Parses local .txt files

Automatic encoding detection

Web Scraper

Idempotent scraper using BeautifulSoup4

Removes boilerplate (<script>, <style>, etc.)

Retry logic with exponential backoff

2. Processing & Embedding Layer

Overlapping Chunking

Sliding window strategy

Chunk Size: 200

Overlap: 50

Local Inference

Model: all-MiniLM-L6-v2

Framework: PyTorch

Runs entirely on CPU/GPU locally

3. Vector Storage Layer

ChromaDB

Persistent, serverless vector database

Atomic Indexing

SHA-256 hashing for deterministic chunk_id

Prevents duplicates

Enables efficient re-indexing

4. Retrieval & Generation

Semantic Search

Cosine similarity

Top-K relevant chunks retrieval

Confidence Gate

Similarity threshold: 0.40

Prevents low-relevance hallucinations

Augmented Synthesis

Strict instruction-based prompting

Uses gpt-4o-mini for grounded responses

🛠️ Tech Stack
Component	Technology
LLM	GPT-4o-mini (GitHub Models via Azure AI Inference SDK)
Embeddings	all-MiniLM-L6-v2 (Local, PyTorch)
Vector DB	ChromaDB (Persistent)
Web Scraping	BeautifulSoup4, Requests
Data Handling	Pandas
Runtime	Python 3.10+
🚀 Installation & Setup
1. Clone Repository & Create Environment
git clone <your-repo-link>
cd local-rag-app
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

2. Install Dependencies
pip install chromadb sentence-transformers azure-ai-inference \
            python-dotenv beautifulsoup4 requests torch

3. API Configuration

Create a .env file in the root directory:

# Get your token from https://github.com/marketplace/models
GITHUB_TOKEN=your_github_personal_access_token

📖 Usage Guide
Step 1: Prepare Knowledge Base

Place .txt files in the documents/ directory

Add URLs (one per line) to:

documents/urls.txt

Step 2: Indexing (Vectorization)

Generate local embeddings and populate ChromaDB:

python app.py --index

Step 3: Interactive Q&A

Start the RAG-powered chat interface:

python app.py
