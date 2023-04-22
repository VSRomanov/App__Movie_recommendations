"""
This file contains the unit tests for the 'movie_recommendations.py' script. It tests the functionality of the
'get_recommendations' function and uses Pearson correlation model to provide movie recommendations based on a given movie.

The tests in this file use the pytest testing framework. To run the tests, simply run the command 'pytest' in the
terminal while in the same directory as this file and the 'movie_recommendations.py' script.

Each test function tests a specific functionality of the 'get_recommendations' function, with different input parameters
and expected output.

Note that the tests require the pre-trained models and dataframes stored in the 'PKL_DIR' directory as '.pkl' files.
Make sure that the 'PKL_DIR' constant is correctly set to the directory where the '.pkl' files are stored before running
the tests.
"""

import pickle

import pandas as pd

from movie_recommend.constants import PKL_DIR
from movie_recommend.utils.get_recommendations import get_recommendations


def test_movie_recommendations_corr():
    # Settings: movie title, number of movies to recommend, model type
    movie_to_compare = "Terminator, The (1984)"
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

    assert first_line == 'Recommendations for "Terminator, The (1984)":'
    assert isinstance(final_table, pd.DataFrame) == True
    assert (
            len(final_table) == num_recommendations
    ), f"Expected {num_recommendations} recommendations, got {len(final_table)}"

    assert (
            movie_to_compare not in final_table.index
    ), f"Expected {movie_to_compare} to be excluded from recommendations"
