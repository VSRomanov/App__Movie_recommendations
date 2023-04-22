"""
This script contains a unit test function to test the mean_rating_table function. The mean_rating_table function computes
the mean rating and total rating count for each movie in the given DataFrame.

The script imports the pandas and numpy modules and defines a fixture that creates a sample ratings DataFrame. The test
function calculates the expected output based on the given sample ratings DataFrame, tests the mean_rating_table function
using the sample_ratings fixture, and asserts that the output message and output DataFrame have the correct format and
contain the expected values.

To run the test, execute the test_mean_rating_table function.
"""

import numpy as np
import pandas as pd
import pytest

from movie_recommend.utils.table_formatting import mean_rating_table


@pytest.fixture
def sample_ratings():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3, 3, 4],
            "title": [
                "Toy Story",
                "The Lion King",
                "Toy Story",
                "The Lion King",
                "The Incredibles",
                "The Lion King",
                "Finding Nemo",
                "Toy Story",
            ],
            "rating": [4.5, 3.0, 3.5, 4.0, 5.0, np.nan, np.nan, np.nan],
            "totalRatingCount": [125, 100, 125, 100, 75, 100, 0, 125],
        }
    )


def test_mean_rating_table(sample_ratings):
    expected_output = pd.DataFrame(
        {"mean_rating": [5.0, 4.0, 3.5, np.nan], "totalRatingCount": [75, 125, 100, 0]},
        index=["The Incredibles", "Toy Story", "The Lion King", "Finding Nemo"],
    )
    expected_output.index.name = "title"

    final_table = mean_rating_table(sample_ratings)

    assert final_table.equals(expected_output)

    assert (
            all(list(final_table["mean_rating"].astype(float))) > 0
    ), "All recommended movies should have a positive rating"
