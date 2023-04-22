"""
This script contains a unit test function to test the filter_movies_by_rating_count function in the table_formatting module.
The filter_movies_by_rating_count function filters out movies from the input DataFrame that have fewer than a certain
number of ratings.

The script imports the pandas and numpy modules and defines a fixture that creates a sample ratings DataFrame. The
test function calculates the expected output based on the given sample ratings DataFrame, tests the filter_movies_by_rating_count
function using the rating_with_totalRatingCount fixture, and asserts that the output DataFrame has the correct
format and contains the expected values.

To run the test, execute the test_filter_movies_by_rating_count function.
"""

import numpy as np
import pandas as pd
import pytest

from movie_recommend.utils.table_formatting import filter_movies_by_rating_count


@pytest.fixture
def rating_with_totalRatingCount():
    return pd.DataFrame(
        {
            "movieId": [
                "1",
                "1",
                "1",
                "2",
                "2",
                "3",
                "3",
                "3",
                "3",
                "4",
                "5",
                "5",
                "5",
            ],
            "title": [
                "Toy Story",
                "Toy Story",
                "Toy Story",
                "The Lion King",
                "The Lion King",
                "The Incredibles",
                "The Incredibles",
                "The Incredibles",
                "The Incredibles",
                "Finding Nemo",
                "The Godfather",
                "The Godfather",
                "The Godfather",
            ],
            "genre": [
                "Animation",
                "Animation",
                "Animation",
                "Adventure",
                "Adventure",
                "Action",
                "Action",
                "Action",
                "Action",
                "Animation",
                "Drama",
                "Drama",
                "Drama",
            ],
            "userId": ["1", "2", "3", "1", "3", "2", "3", "4", "5", "5", "2", "3", "5"],
            "rating": [
                5.0,
                4.0,
                3.5,
                5.0,
                4.5,
                4.5,
                4.0,
                3.5,
                4.0,
                3.5,
                np.nan,
                np.nan,
                np.nan,
            ],
            "totalRatingCount": [3, 3, 3, 2, 2, 4, 4, 4, 4, 1, 0, 0, 0],
        }
    )


def test_filter_movies_by_rating_count(rating_with_totalRatingCount):
    num_rating_threshold = 2
    expected_output = pd.DataFrame(
        {
            "movieId": ["1", "1", "1", "3", "3", "3", "3"],
            "title": [
                "Toy Story",
                "Toy Story",
                "Toy Story",
                "The Incredibles",
                "The Incredibles",
                "The Incredibles",
                "The Incredibles",
            ],
            "genre": [
                "Animation",
                "Animation",
                "Animation",
                "Action",
                "Action",
                "Action",
                "Action",
            ],
            "userId": ["1", "2", "3", "2", "3", "4", "5"],
            "rating": [5.0, 4.0, 3.5, 4.5, 4.0, 3.5, 4.0],
            "totalRatingCount": [3, 3, 3, 4, 4, 4, 4],
        }
    )

    actual_output = filter_movies_by_rating_count(
        rating_with_totalRatingCount, num_rating_threshold
    )

    assert actual_output.equals(expected_output)
