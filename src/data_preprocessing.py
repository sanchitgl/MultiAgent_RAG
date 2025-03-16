import pandas as pd

def load_and_merge_data(ratings_path, mpst_path, titles_path):
    """
    Loads ratings, mpst, and titles data, merges them into a final DataFrame.
    Returns the merged DataFrame with essential columns (including 'plot_synopsis').
    """
    
    ratings_df = pd.read_csv(ratings_path, sep="\t")
    mpst_df = pd.read_csv(mpst_path)

    merged_df = ratings_df.merge(mpst_df, left_on="tconst", right_on="imdb_id", how="left")
    notna_merged_df = merged_df[merged_df['imdb_id'].notnull()]

    titles_df = pd.read_csv(titles_path, sep="\t")

    movie_df = notna_merged_df.sample(100, random_state=42)

    final_df = movie_df.merge(titles_df, left_on="imdb_id", right_on="tconst", how="left")

    final_df = final_df[['imdb_id', 'title', 'plot_synopsis', 'titleType']].dropna(subset=['plot_synopsis'])

    return final_df