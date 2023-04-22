from typing import List, Tuple

import pandas as pd
import sklearn
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def recommendation_rename_movie(
    movie_to_compare: str,
    movie_array: List[str],
    num_recommendations: int,
    total_ratings: pd.DataFrame,
) -> Tuple[str, pd.DataFrame]:
    """
    Recommends movies based on a partial match to the given movie title and provides
    similarity scores, total rating count and mean rating per movie.

    Args:
        movie_to_compare (str): The title of the movie to compare against.
        movie_array (list): A list of movie titles to search for partial matches.
        num_recommendations (int): The number of recommendations to generate.
        total_ratings (pandas.DataFrame): A DataFrame with the total rating count and mean rating per movie.

    Returns:
        tuple: A tuple containing the first line of the response and a DataFrame with the recommendations.
    """
    # Suggest alternative titles based on partial matching
    message = (
        f'No "{movie_to_compare}" movie in the database. Try the following titles:'
    )

    # Remove the year at the end of the movie title (to improve suggestions of alternatives)
    movie_to_compare = movie_to_compare.rstrip("(0123456789)").rstrip()

    # Get all titles with similarity score and return the top recommendations
    table_all = pd.DataFrame(data=movie_array, columns=["title"])
    table_all["similarity_score"] = table_all["title"].apply(
        lambda x: fuzz.partial_ratio(movie_to_compare, x)
    )
    table = (
        table_all.sort_values("similarity_score", ascending=False)
        .reset_index(drop=True)
        .head(num_recommendations)
    )

    table = pd.merge(table, total_ratings, on="title")

    table = table.reindex(
        columns=["title", "mean_rating", "totalRatingCount", "similarity_score"]
    )

    return message, table


def knn_train(
    movie_features_df: pd.DataFrame,
) -> sklearn.neighbors._unsupervised.NearestNeighbors:
    """
    Trains a k-Nearest Neighbors model on a given dataset of movie features using the cosine distance metric.

    Args:
        movie_features_df (pandas.DataFrame): A DataFrame with movie features as columns and movie titles as indices.

    Returns:
        sklearn.neighbors._unsupervised.NearestNeighbors: A k-Nearest Neighbors model trained on the movie features.
    """
    # Table as matrix
    movie_features_df_matrix = csr_matrix(movie_features_df.values)

    # k-NearestNeighbors model
    model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
    model_knn.fit(movie_features_df_matrix)

    return model_knn


def recommendation_knn(
    movie_features_df: pd.DataFrame,
    model: NearestNeighbors,
    movie_to_compare: str,
    num_recommendations: int,
    total_ratings: pd.DataFrame,
) -> Tuple[str, pd.DataFrame]:
    """
    Recommends similar movies to a given movie using k-Nearest Neighbors algorithm.

    Args:
        movie_features_df (pandas.DataFrame): A DataFrame with movie features as columns and movie titles as indices.
        model (sklearn.neighbors._unsupervised.KNNUnsupervised): A trained k-Nearest Neighbors model.
        movie_to_compare (str): The title of the movie to find similar movies for.
        num_recommendations (int): The number of recommendations to generate.
        total_ratings (pd.DataFrame): A DataFrame with the total rating count and mean rating per movie.

    Returns:
        tuple: A tuple containing the first line of the response and a DataFrame with the recommendations.
    """
    # Get the index of the movie to compare
    movie_index = movie_features_df.index.get_loc(movie_to_compare)

    # Using 'model', calculate the distances and indices of the k-Nearest Neighbors relative to 'movie_index'
    distances, indices = model.kneighbors(
        movie_features_df.iloc[movie_index, :].values.reshape(1, -1),
        n_neighbors=num_recommendations + 1,
    )

    # Create a DataFrame containing the recommendations
    table = pd.DataFrame(columns=["title", "distance"])
    for i in range(0, len(distances.flatten())):
        if i == 0:
            # If this is the first recommendation, get a header message
            message = f'Recommendations for "{movie_to_compare}":'
        else:
            # Otherwise, add the recommendation to the DataFrame
            new_row = pd.DataFrame(
                [
                    [
                        movie_features_df.index[indices.flatten()[i]],
                        distances.flatten()[i],
                    ]
                ],
                columns=["title", "distance"],
            )
            table = pd.concat([table, new_row], axis=0, ignore_index=True)

    table = table.join(total_ratings, on="title")
    table = table.reindex(
        columns=["title", "mean_rating", "totalRatingCount", "distance"]
    )

    return message, table


def recommendation_corr(
    movie_features_df: pd.DataFrame,
    movie_to_compare: str,
    num_recommendations: int,
    total_ratings: pd.DataFrame,
) -> Tuple[str, pd.DataFrame]:
    """
    Recommends top movies based on the Pearson correlation between a specified movie and other movies in the dataset.

    Args:
        movie_features_df (pandas.DataFrame): A dataframe containing movie features.
        movie_to_compare (str): The title of the movie to compare against.
        num_recommendations (int): The number of recommendations to generate.
        total_ratings (pd.DataFrame): A DataFrame with the total rating count and mean rating per movie.

    Returns:
        tuple: A tuple containing the header message (str) and a dataframe containing the recommendations (pandas.DataFrame).
    """

    # Drop rows with NaN values in the specified movie column
    movie_features_df__nonan = movie_features_df.dropna(subset=[movie_to_compare])

    # Calculate Pearson correlations between 'movie_to_compare' and other movies
    correlations = (
        movie_features_df__nonan[movie_to_compare].corr(
            movie_features_df__nonan[col], min_periods=100
        )
        for col in movie_features_df__nonan
    )

    # Combine correlations with column names and sort by correlation coefficient
    corr_to_my_movie = pd.DataFrame(
        {"title": movie_features_df__nonan.columns, "correlation": correlations}
    )
    corr_to_my_movie.sort_values(by="correlation", ascending=False, inplace=True)

    # Set the index of the DataFrame to the title column
    corr_to_my_movie.set_index("title", inplace=True)

    # Drop the movie being compared from the recommendations and reset the index
    corr_to_my_movie.drop(movie_to_compare, inplace=True)
    corr_to_my_movie.reset_index(inplace=True)

    # Join the correlation DataFrame with the total_ratings DataFrame and rearrange the columns
    table = corr_to_my_movie.iloc[:num_recommendations].join(total_ratings, on="title")
    table = table.reindex(
        columns=["title", "mean_rating", "totalRatingCount", "correlation"]
    )

    # A header message for the recommendations
    message = f'Recommendations for "{movie_to_compare}":'

    return message, table
