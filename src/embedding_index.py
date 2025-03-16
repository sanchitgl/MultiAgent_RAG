import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_and_save_index(movie_df, index_path, ids_path, model_name='all-MiniLM-L6-v2'):
    """
    Builds a FAISS HNSW index from the movie DataFrame's plot synopses, 
    then saves the index and the corresponding imdb_ids.
    """
    print("Generating embeddings...")
    model = SentenceTransformer(model_name)

    # Encode plot synopses
    plot_embeddings = model.encode(
        movie_df['plot_synopsis'].tolist(),
        batch_size=32,
        convert_to_tensor=False
    )
    plot_embeddings = np.array(plot_embeddings).astype('float32')

    # Build the FAISS HNSW index
    embedding_dim = plot_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(embedding_dim, 32)  # 32 neighbor links
    index.hnsw.efConstruction = 200

    # Add embeddings
    index.add(plot_embeddings)

    # Save index and ID mappings
    faiss.write_index(index, index_path)
    np.save(ids_path, movie_df['imdb_id'].values)

    print("Index built and saved!")

def load_index(index_path):
    """Loads and returns a FAISS index from the given path."""
    return faiss.read_index(index_path)