"""
This script contains a unit test function to test the recommendation_rename_movie function. The recommendation_rename_movie
function takes a movie name and returns the top recommended movies based on similarity score.

The script imports numpy, pandas, and pytest modules, as well as assert_frame_equal from the pandas._testing module.
It defines a fixture that creates a sample DataFrame containing movie titles, mean ratings, and total rating counts.
The test function test_recommendation_rename_movie takes the fixture and tests the recommendation_rename_movie function
for a specific movie name.

The function asserts that the output message and output DataFrame have the correct format and contain the expected
values. The output message should be a string that contains the movie name and recommended movies. The output DataFrame
should contain the recommended movie titles, mean ratings, total rating counts, and similarity scores.

To run the test, execute the test_recommendation_rename_movie function.
"""

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from movie_recommend.utils.recommendation_algorithms import recommendation_rename_movie


@pytest.fixture
def sample_total_ratings():
    return pd.DataFrame(
        {
            "title": [
                "Toy Story",
                "Toy Story 2",
                "The Lion King",
                "The Incredibles",
                "Finding Nemo",
            ],
            "mean_rating": [4.0, np.nan, 3.5, 5.0, 3.0],
            "totalRatingCount": [125, np.nan, 100, 75, 0],
        }
    )


def test_recommendation_rename_movie(sample_total_ratings):
    movie_to_compare = "Toy Storyy"
    movie_array = [
        "Toy Story",
        "Toy Story 2",
        "The Lion King",
        "The Incredibles",
        "Finding Nemo",
    ]
    num_recommendations = 3
    expected_message = (
        'No "Toy Storyy" movie in the database. Try the following titles:'
    )
    expected_table = pd.DataFrame(
        {
            "title": ["Toy Story", "Toy Story 2", "The Lion King"],
            "mean_rating": [4.0, np.nan, 3.5],
            "totalRatingCount": [125, np.nan, 100],
            "similarity_score": [100, 90, 30],
        }
    )

    result = recommendation_rename_movie(
        movie_to_compare, movie_array, num_recommendations, sample_total_ratings
    )
    result_message = result[0]
    result_table = result[1]

    # Assert that the output message is correct
    assert result_message == expected_message

    # Assert that the output table is correct
    assert isinstance(result_table, pd.DataFrame) == True
    assert_frame_equal(result_table, expected_table, check_dtype=True)
    assert (
            len(result_table) == num_recommendations
    ), f"Expected {num_recommendations} recommendations, got {len(result_table)}"
