# RAG-based Movie Recommendation System with FAISS HNSW

This project implements a **Retrieval-Augmented Generation (RAG)** based **movie recommendation system** that utilizes **FAISS HNSW** for efficient similarity search. The system generates **SBERT embeddings** for movie plots and indexes them with FAISS to retrieve the most similar movies based on content similarity.

---

## Key Features

- **Retrieval-Augmented Generation (RAG)**: Enhances movie recommendations by retrieving similar content before generating results.
- **Efficient FAISS HNSW Indexing**: Uses Hierarchical Navigable Small World (HNSW) graphs for fast approximate nearest neighbor (ANN) search.
- **Sentence-BERT (SBERT) Embeddings**: Generates high-quality embeddings from movie plot synopses.
- **Scalable Similarity Search**: Supports large datasets with efficient memory usage and retrieval times.

---

## Installation

### Clone the Repository

```bash
git clone <repo-url>
cd movie_recommender_project
```

### Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Usage

The project supports three primary tasks:

### 1. Preprocess Data

```bash
python main.py --task preprocess
```

Processes and saves the dataset for indexing.

### 2. Build FAISS HNSW Index

```bash
python main.py --task build_index
```

This generates the FAISS index for efficient movie retrieval.

### 3. Retrieve Similar Movies Using FAISS

```bash
python main.py --task recommend --query "Hotel Transylvania" --top_k 5
```

Replace `"Hotel Transylvania"` with any movie title to get the top-K similar movies based on plot similarity.


### Example Output



<img width="424" alt="Screenshot 2025-03-17 at 11 09 54 AM" src="https://github.com/user-attachments/assets/c58fd916-805c-4870-9780-f9007c70824b" />

```bash
Recommended Movies: [('House of Frankenstein', 0.74598217), ('Bud Abbott Lou Costello Meet Frankenstein', 0.85303336), ('Love at First Bite', 0.89539623), ('Billy the Kid Versus Dracula', 0.90585554), ('Cube 2: Hypercube', 0.90817416), ('The Return of the Vampire', 0.9284278), ('Monsters University', 0.93450934), ('Friday the 13th Part III', 0.9452752), ('Dracula: The Musical', 0.9492428), ('Count Dracula', 0.9529716)]​
```


## 📄 License

This project is licensed under the MIT License.
