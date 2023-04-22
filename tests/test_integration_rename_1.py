"""
This file contains a unit test for the 'get_recommendations' function in the 'get_recommendations.py' script. The purpose
of this unit test is to verify that the 'get_recommendations' function can accurately recommend movies based on
similarity scores even when the provided movie title is not present in the database.

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


def test_movie_recommendations_rename_1():
    # Settings: movie title, number of movies to recommend, model type
    movie_to_compare = "Terminator"
    num_recommendations = 20
    model_type = "corr"

    # Load dataframe for Pearson correlation algorithm from .pkl file
    with open(PKL_DIR / "corr_model.pkl", "rb") as f:
        try:
            movie_features_df__corr, total_ratings, total_movie_array = pickle.load(f)
        except Exception as e:
            assert False, f"Error loading corr_model: {e}"

    movie_features_df = movie_features_df__corr
    model = "none"

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
            first_line == 'No "Terminator" movie in the database. Try the following titles:'
    )
    assert isinstance(final_table, pd.DataFrame) == True
    assert (
            len(final_table) == num_recommendations
    ), f"Expected {num_recommendations} recommendations, got {len(final_table)}"

    assert (
            movie_to_compare not in final_table.index
    ), f"Expected {movie_to_compare} to be excluded from recommendations"
