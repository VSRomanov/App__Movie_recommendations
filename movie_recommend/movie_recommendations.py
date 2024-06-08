"""
Script to get movie recommendations using a K-Nearest Neighbors (KNN) or Pearson correlation model.

Local usage:
- Modify the 'movie_to_compare' variable to specify the movie title to compare.
- Modify the 'num_recommendations' variable to specify the number of recommended movies.
- Uncomment the desired 'model_type' value (either 'knn' or 'corr').
- Run the script to get the recommendations.

The script loads the pre-trained KNN model and pre-formatted dataframes from pkl files. It determines
which model and features to use based on the 'model_type' value. It then calls the 'get_recommendations()'
function from the 'movie_recommend.app' module to get the movie recommendations using the selected model
and features.
"""

import os
import pickle
from typing import Tuple

import logging
import pandas as pd

from movie_recommend.utils.get_recommendations import get_recommendations
import movie_recommend.constants as c

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_pkl_file_name(model_type: str, db_size: str) -> str:
    return f"{model_type}_model_{db_size}.pkl"

def load_data_from_pkl(pkl_file: str) -> Tuple:
    """Load data from pkl file."""
    try:
        with open(pkl_file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        exit()

class MovieRecommend:
    def __init__(self, model_type: str, db_size: str, n_recommend: int = 20):
        self.model_type = model_type
        self.db_size = db_size
        self.n_recommend = n_recommend

    def launch(self, movie_to_compare: str) -> Tuple[str, pd.DataFrame]:
        pkl_file = get_pkl_file_name(self.model_type, self.db_size)
        features_df, model, all_ratings, total_movie_array = load_data_from_pkl(os.path.join(c.PKL_DIR, pkl_file))
        movie_array = features_df.columns

        # Get the movie recommendations
        first_line, final_table = get_recommendations(
            features_df, movie_to_compare, self.n_recommend, self.model_type, model, all_ratings,
            total_movie_array
        )

        logging.info(
            f"Number of movies to compare with (number of ratings is higher than the threshold): {len(movie_array)}"
        )

        return first_line, final_table


if __name__ == "__main__":
    # Settings: movie title, number of movies to recommend, model type
    # movie_to_compare = 'Jumanji (1995)'
    # movie_to_compare = 'Commando (1985)'
    # movie_to_compare = "Terminator"
    # movie_to_compare = "The Terminators (2009)"
    # movie_to_compare = "Terminator, The (1984)"
    # movie_to_compare = "1408 (2007)"
    # movie_to_compare = 'Rambo: First Blood Part II (1985)'
    # movie_to_compare = 'Rambo: First Blood (1982)'
    # movie_to_compare = "First Blood (Rambo: First Blood) (1982)"
    # movie_to_compare = 'Avatar (2009)'
    movie_to_compare = "Star Wars"
    # movie_to_compare = "Robot Chicken: Star Wars (2007)"
    # movie_to_compare = "Star Trek (2009)"
    # movie_to_compare = 'Star Wars: Episode VI - Return of the Jedi (1983)'
    # movie_to_compare = 'Puss in Boots (2011)'
    # movie_to_compare = "Brother (Brat) (1997)"
    # movie_to_compare = "Brat"
    # movie_to_compare = "Exterminator"
    # movie_to_compare = "Operation 'Y' & Other Shurik's Adventures (1965)"

    # Configuration
    # model_type = "knn"
    model_type = "corr"
    # dataset_size = "small"
    dataset_size = "full"

    first_line, final_table = MovieRecommend(model_type=model_type, db_size=dataset_size, n_recommend=20).launch(movie_to_compare=movie_to_compare)

    # Set pandas options for wider print-out
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)

    print(first_line)
    print(final_table)