import argparse 
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_preprocessing import load_and_merge_data

from src.embedding_index import build_and_save_index, load_index
from src.recommend import recommend_movies

def main():
    parser = argparse.ArgumentParser(description="Movie Recommender System")
    parser.add_argument("--task", type=str, required=True, choices=["preprocess", "build_index", "recommend"],
                        help="Task to perform: preprocess, build_index, or recommend")
    parser.add_argument("--query", type=str, help="Movie title for recommendation (if using --task recommend)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations (default: 10)")

    args = parser.parse_args()

    # Define file paths
    ratings_path = os.path.join("data", "title_ratings_sample.tsv")
    mpst_path = os.path.join("data", "mpst_full_data.csv")
    titles_path = os.path.join("data", "title_basics_sample.tsv")
    index_path = "movie_hnsw.index"
    ids_path = "movie_ids.npy"

    if args.task == "preprocess":
        print("Preprocessing data...")
        final_df = load_and_merge_data(ratings_path, mpst_path, titles_path)
        final_df.to_csv("data/processed_movies.csv", index=False)
        print("Preprocessed data saved to 'processed_movies.csv'")

    elif args.task == "build_index":
        print("Building FAISS index...")
        final_df = load_and_merge_data(ratings_path, mpst_path, titles_path)
        build_and_save_index(final_df, index_path, ids_path)
        print("Indexing complete.")

    elif args.task == "recommend":
        if not args.query:
            print("Error: You must provide a movie title using --query")
            return
        
        # Load data & index
        print("Loading preprocessed data...")
        final_df = pd.read_csv("data/processed_movies.csv")
        index = load_index(index_path)
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Get recommendations
        recommendations = recommend_movies(args.query, final_df, index, model, args.top_k)
        print(recommendations)
        print("\nRecommended Movies:")
        for title, score in recommendations:
            print(f"- {title} (Score: {score:.4f})")

if __name__ == "__main__":
    main()