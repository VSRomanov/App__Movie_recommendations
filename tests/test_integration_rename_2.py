"""
This file contains a unit test for the 'get_recommendations' function in the 'get_recommendations.py' script. The purpose
of this unit test is to verify that the 'get_recommendations' function can recommend other movies based on
similarity scores when the number of ratings for selected movie is not enough for the analysis.

The test sets the parameters for the movie to compare, number of recommendations, and model type and loads a pre-trained
dataframe for the algorithm. The function is then called and the output is tested against expected results for various
assertions, including the correct output message, expected number of recommendations and the exclusion of the input movie
from the recommendations.

This test uses the pytest testing framework and requires pre-trained models and dataframes stored in the 'PKL_DIR'
directory as '.pkl' files. Before running the tests, ensure that the 'PKL_DIR' constant is correctly set to the
directory where the '.pkl' files are stored.
"""

import pickle

import pandas as pd

from movie_recommend.constants import PKL_DIR
from movie_recommend.utils.get_recommendations import get_recommendations


def test_movie_recommendations_rename_2():
    # Settings: movie title, number of movies to recommend, model type
    movie_to_compare = "Exterminator 2 (1984)"
    num_recommendations = 20
    model_type = "knn"

    # Load KNN model from .pkl file
    with open(PKL_DIR / "knn_model.pkl", "rb") as f:
        try:
            (
                movie_features_df__knn,
                model_knn,
                total_ratings,
                total_movie_array,
            ) = pickle.load(f)
        except Exception as e:
            assert False, f"Error loading knn_model: {e}"

    movie_features_df = movie_features_df__knn
    model = model_knn

    # Get the movie recommendations
    first_line, final_table = get_recommendations(
        movie_features_df,
        movie_to_compare,
        num_recommendations,
        model_type,
        model,
        total_ratings,
        total_movie_array,
    )

    assert (
            first_line
            == 'Number of ratings for "Exterminator 2 (1984)" is not enough for the analysis. Try another movie.\n'
    )
    assert isinstance(final_table, pd.DataFrame) == True
    assert (
            len(final_table) == num_recommendations
    ), f"Expected {num_recommendations} recommendations, got {len(final_table)}"

    assert (
            movie_to_compare not in final_table.index
    ), f"Expected {movie_to_compare} to be excluded from recommendations"
