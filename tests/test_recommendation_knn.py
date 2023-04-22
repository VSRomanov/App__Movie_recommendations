"""
This script contains a unit test function to test the recommendation_knn function. The recommendation_knn function
uses a k-Nearest Neighbors algorithm to make movie recommendations based on a user's selected movie.

The script imports pandas and sklearn.neighbors modules, defines a fixture that creates a sample movie features
DataFrame and a sample ratings DataFrame, trains a k-Nearest Neighbors model on the movie features, and tests the
recommendation_knn function for a specific movie. The function asserts that the output message and output DataFrame
have the correct format and contain the expected values.

To run the test, execute the test_recommendation_knn function.
"""

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from movie_recommend.utils.recommendation_algorithms import recommendation_knn


@pytest.fixture
def sample_data():
    # A movie for correlation
    movie_to_compare = "The Shawshank Redemption"

    # Define the size of output
    num_recommendations = 6

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
    data_transposed_no_na = data_transposed.fillna(0)

    # Table as matrix
    movie_features_df_matrix = csr_matrix(data_transposed_no_na.values)

    # Train a k-nearest neighbors model
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(movie_features_df_matrix)

    # Get the index of the movie to compare
    movie_index = data_transposed_no_na.index.get_loc(movie_to_compare)

    # Calculate the distances and indices of the k-nearest neighbors relative to 'movie_index'
    distances, indices = model.kneighbors(
        data_transposed_no_na.iloc[movie_index, :].values.reshape(1, -1),
        n_neighbors=num_recommendations + 1,
    )

    data_transposed["mean_rating"] = data_transposed.mean(axis=1, skipna=True).astype(
        float
    )
    data_transposed["totalRatingCount"] = (
        data_transposed.iloc[:, :400].count(axis=1, numeric_only=True).astype(int)
    )
    total_ratings = data_transposed.iloc[:, -2:]
    # print(total_ratings)

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
                        total_ratings.index[indices.flatten()[i]],
                        distances.flatten()[i],
                    ]
                ],
                columns=["title", "distance"],
            )
            table = pd.concat([table, new_row], axis=0, ignore_index=True)

    table = table.join(total_ratings, on="title")

    # Round the expected correlation values for comparison
    table.sort_values(by="distance", ascending=True, inplace=True)

    expected_output = table.loc[
        :, ["title", "mean_rating", "totalRatingCount", "distance"]
    ]
    # print(expected_output)

    return (
        data_transposed_no_na,
        model,
        movie_to_compare,
        total_ratings,
        expected_output,
        num_recommendations,
        message,
    )


def test_recommendation_knn(sample_data):
    # Unpack the sample data
    (
        movie_features_df,
        model,
        movie_to_compare,
        total_ratings,
        expected_output,
        num_recommendations,
        message,
    ) = sample_data

    # Call the function to get the actual output
    message_output, actual_output = recommendation_knn(
        movie_features_df, model, movie_to_compare, num_recommendations, total_ratings
    )

    # Assert that the output message is correct
    assert message_output == message

    # Assert that the output DataFrame has the correct number of rows and columns
    assert isinstance(actual_output, pd.DataFrame) == True
    assert actual_output.shape == (6, 4)

    # Check that the actual output matches the expected output
    assert_frame_equal(actual_output, expected_output, check_dtype=True)
