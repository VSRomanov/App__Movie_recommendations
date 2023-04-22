"""
This script contains a unit test function to test the recommendation_corr function. The recommendation_corr function
uses Pearson correlation to make movie recommendations based on a user's selected movie.

The script imports the pandas and numpy modules and defines a fixture that creates a sample movie features DataFrame
and a sample ratings DataFrame. The test function calculates the expected output based on Pearson correlation, tests
the recommendation_corr function for a specific movie, and asserts that the output message and output DataFrame have
the correct format and contain the expected values.

To run the test, execute the test_recommendation_corr function.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from movie_recommend.utils.recommendation_algorithms import recommendation_corr


@pytest.fixture
def sample_data():
    # A movie for correlation
    movie_to_compare = "The Shawshank Redemption"

    # create sample data with NaNs and 7000 columns
    num_cols = 7
    num_rows = 400
    movie_array = [
        "The Shawshank Redemption",
        "The Godfather",
        "The Dark Knight",
        "Forrest Gump",
        "Inception",
        "Titanic",
        "Pulp Fiction",
    ]

    # Set the seed value
    np.random.seed(41)

    movie_features_df = pd.DataFrame(
        np.random.randint(1, 6, size=(num_rows, num_cols)), columns=movie_array
    )
    movie_features_df.iloc[0:10, 0] = np.nan
    movie_features_df.iloc[10:400, 1] = np.nan
    movie_features_df.iloc[15:400, 2] = np.nan

    data_transposed = movie_features_df.transpose()
    data_transposed["mean_rating"] = data_transposed.mean(axis=1, skipna=True).astype(
        float
    )
    data_transposed["totalRatingCount"] = (
        data_transposed.iloc[:, :400].count(axis=1, numeric_only=True).astype(int)
    )

    total_ratings = data_transposed.iloc[:, -2:]
    # print(total_ratings)

    # Round the expected correlation values for comparison
    data_transposed["correlation"] = movie_features_df.corr(min_periods=100)[
        movie_to_compare
    ]
    data_transposed.sort_values(by="correlation", ascending=False, inplace=True)
    data_transposed = data_transposed.iloc[1:]
    data_transposed.reset_index(inplace=True)
    data_transposed = data_transposed.rename(columns={"index": "title"})

    expected_output = data_transposed.loc[
                      :, ["title", "mean_rating", "totalRatingCount", "correlation"]
                      ]
    # print(expected_output)

    return movie_features_df, movie_to_compare, total_ratings, expected_output


def test_recommendation_corr(sample_data):
    # Unpack the sample data
    movie_features_df, movie_to_compare, total_ratings, expected_output = sample_data

    # Define the size of output
    num_recommendations = 20

    # Call the function to get the actual output
    message, actual_output = recommendation_corr(
        movie_features_df, movie_to_compare, num_recommendations, total_ratings
    )

    # Assert that the output message is correct
    assert message == 'Recommendations for "The Shawshank Redemption":'

    # Assert that the output DataFrame has the correct number of rows and columns
    assert isinstance(actual_output, pd.DataFrame) == True
    assert actual_output.shape == (6, 4)

    # Check that the actual output matches the expected output
    assert_frame_equal(actual_output, expected_output, check_dtype=True)
