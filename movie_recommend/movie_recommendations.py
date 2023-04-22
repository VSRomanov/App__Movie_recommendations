"""
Script to get movie recommendations using a K-Nearest Neighbors (KNN) or Pearson correlation model.

Usage:
- Modify the 'movie_to_compare' variable to specify the movie title to compare.
- Modify the 'num_recommendations' variable to specify the number of recommended movies.
- Uncomment the desired 'model_type' value (either 'knn' or 'corr').
- Run the script to get the recommendations.

The script loads the pre-trained KNN model and pre-formatted dataframes from .pkl files. It determines
which model and features to use based on the 'model_type' value. It then calls the 'get_recommendations'
function from the 'movie_recommend.app' module to get the movie recommendations using the selected model
and features.

"""

import pickle

import pandas as pd

from movie_recommend.app import get_recommendations
from movie_recommend.constants import PKL_DIR

if __name__ == "__main__":
    # Settings: movie title, number of movies to recommend, model type
    # movie_to_compare = 'Jumanji (1995)'
    # movie_to_compare = 'Commando (1985)'
    # movie_to_compare = "Terminator"
    # movie_to_compare = "The Terminators (2009)"
    # movie_to_compare = "Terminator, The (1984)"
    # movie_to_compare = 'Rambo: First Blood Part II (1985)'
    # movie_to_compare = 'Rambo: First Blood (1982)'
    # movie_to_compare = "First Blood (Rambo: First Blood) (1982)"
    # movie_to_compare = 'Avatar (2009)'
    # movie_to_compare = 'Star Wars'
    # movie_to_compare = "Robot Chicken: Star Wars (2007)"
    # movie_to_compare = 'Star Wars: Episode VI - Return of the Jedi (1983)'
    # movie_to_compare = 'Puss in Boots (2011)'
    # movie_to_compare = "Brother (Brat) (1997)"
    movie_to_compare = "Brat"
    # movie_to_compare = "Exterminator"
    # movie_to_compare = "Operation 'Y' & Other Shurik's Adventures (1965)"

    num_recommendations = 20

    model_type = "knn"
    # model_type = "corr"

    # Load KNN model from .pkl file
    with open(PKL_DIR / "knn_model.pkl", "rb") as f:
        try:
            (
                movie_features_df__knn,
                model_knn,
                total_ratings,
                total_movie_array,
            ) = pickle.load(f)
        except Exception as e:
            print(f"Error loading knn_model: {e}")
            exit()

    # Load dataframe for Pearson correlation algorithm from .pkl file
    with open(PKL_DIR / "corr_model.pkl", "rb") as f:
        try:
            movie_features_df__corr, total_ratings, total_movie_array = pickle.load(f)
        except Exception as e:
            print(f"Error loading corr_model: {e}")
            exit()

    # Determine which model and features to use based on the model type
    if model_type == "knn":
        movie_features_df = movie_features_df__knn
        model = model_knn
        movie_array = movie_features_df.index
    elif model_type == "corr":
        movie_features_df = movie_features_df__corr
        model = "none"
        movie_array = movie_features_df.columns

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

    print(
        f"\nNumber of movies to compare with (with number of ratings higher than the threshold): {len(movie_array)}\n"
    )

    pd.set_option("display.expand_frame_repr", False)  ## Wider print-out
    print(first_line)
    print(final_table)
