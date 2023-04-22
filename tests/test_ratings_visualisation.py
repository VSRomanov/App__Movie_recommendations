"""
This pytest script contains a unit test function to test the ratings_visualisation function from the movie_recommend
package. The ratings_visualisation function creates a visualization of the distribution of movie ratings based on some
input parameters.

The test function creates a sample DataFrame of movie ratings data, passes it to the ratings_visualisation function
along with some input parameters, and checks that the output file was created and is not empty. The temporary output
folder is then deleted to clean up after the test.

To run the test, execute the test_ratings_visualisation function. This test helps to ensure that the ratings_visualisation
function is working as expected and produces the desired output, given a specific input.

"""

import shutil

import pandas as pd

from movie_recommend.constants import OUTPUT_DIR
from movie_recommend.utils.data_visualisation import ratings_visualisation


def test_ratings_visualisation():
    # Create a sample DataFrame with 20 rows
    data = {
        "movie_id": [i for i in range(20)],
        "mean_rating": [
            3.5,
            4.2,
            2.8,
            3.9,
            4.1,
            2.5,
            3.2,
            3.8,
            3.9,
            4.3,
            3.0,
            2.7,
            3.6,
            3.1,
            3.9,
            4.0,
            3.3,
            3.8,
            4.5,
            3.7,
        ],
        "totalRatingCount": [
            100,
            150,
            200,
            50,
            180,
            70,
            110,
            220,
            90,
            120,
            160,
            90,
            100,
            80,
            50,
            120,
            70,
            80,
            100,
            140,
        ],
    }
    df = pd.DataFrame(data)

    # Set up the parameters for the function
    dataset_size = "small"
    num_rating_threshold = 100
    rating_threshold = 3.5
    tmp_path = OUTPUT_DIR / "test_output_fig"

    # Call the function
    ratings_visualisation(
        df, dataset_size, num_rating_threshold, rating_threshold, tmp_path
    )

    output_file_path = tmp_path / f"rating_vs_totalRatingCount_{dataset_size}.png"

    # Assert that the file was created
    assert (tmp_path / output_file_path).is_file()

    # Check that the output file is not empty
    assert output_file_path.stat().st_size > 0

    # Delete temporary output folder
    shutil.rmtree(tmp_path)
