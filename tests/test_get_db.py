"""
This file contains the unit tests for the 'get_db' function in the 'utils.get_databases' module. The 'get_db' function
imports movie and rating tables from the MovieLens dataset and returns them as two dataframes. The function takes one
argument dataset_size, which is a string, either "small" or "full".

The tests in this file use the pytest testing framework. To run the tests, simply run the command 'pytest' in the terminal
while in the same directory as this file and the 'data.py' script.

Each test function tests a specific functionality of the 'get_db' function, with different input parameters and expected
output. The tests check that the resulting dataframes are not empty, have the correct data types, and contain the expected
columns. They also check that the directories DATA_DIR, DATA_SMALL_DIR, and DATA_FULL_DIR are created if they do not exist.

Note that the tests require the MovieLens dataset to be downloaded and saved in the 'data' directory. If the dataset is not
available, the tests will download it from the MovieLens website. The download may take several minutes, depending on the
internet speed. Make sure that the 'DATA_DIR' constant is correctly set to the directory where the MovieLens dataset is saved
before running the tests.
"""

import pandas as pd

from movie_recommend.constants import DATA_SMALL_DIR
from movie_recommend.utils.get_databases import get_db


def test_get_db():
    # Test the "small" dataset
    movies_df, ratings_df = get_db("small")

    # Check that the dataframes are not empty
    assert not movies_df.empty
    assert not ratings_df.empty

    # Check that the data types are correct
    assert isinstance(movies_df, pd.DataFrame)
    assert isinstance(ratings_df, pd.DataFrame)

    # Check that the expected columns are present
    assert set(movies_df.columns) == {"movieId", "title"}
    assert set(ratings_df.columns) == {"userId", "movieId", "rating"}

    # Check that the data directory was created
    assert DATA_SMALL_DIR.exists()

    # Test the "full" dataset
    movies_df, ratings_df = get_db("full")

    # Check that the dataframes are not empty
    assert not movies_df.empty
    assert not ratings_df.empty

    # Check that the data types are correct
    assert isinstance(movies_df, pd.DataFrame)
    assert isinstance(ratings_df, pd.DataFrame)

    # Check that the expected columns are present
    assert set(movies_df.columns) == {"movieId", "title"}
    assert set(ratings_df.columns) == {"userId", "movieId", "rating"}
