import numpy as np
from sentence_transformers import SentenceTransformer

def recommend_movies(query_title, movie_df, index, model, top_k=10):
    """
    Finds the top-k similar movies based on plot synopsis using FAISS.
    """
    query_row = movie_df[movie_df['title'].str.lower() == query_title.lower()]
    if query_row.empty:
        return f"Movie '{query_title}' not found in database."
    
    # Encode query movie's plot
    query_embedding = model.encode(
        query_row.iloc[0]['plot_synopsis']
    ).reshape(1, -1).astype('float32')

    # Search for similar movies
    distances, indices = index.search(query_embedding, top_k + 1)

    # Collect recommendations (excluding first result which is the movie itself)
    recommendations = []
    for i, idx in enumerate(indices[0][1:]):
        rec_movie = movie_df.iloc[idx]['title']
        recommendations.append((rec_movie, distances[0][i+1]))
    
    return recommendations