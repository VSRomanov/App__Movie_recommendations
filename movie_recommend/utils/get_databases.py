from io import BytesIO  # # Keep the unzipped file in memory, not saving into the drive
from typing import Tuple
from zipfile import ZipFile

import pandas as pd
import requests
from pandas import DataFrame

from movie_recommend.constants import DATA_DIR, DATA_FULL_DIR, DATA_SMALL_DIR


def get_db(dataset_size: str) -> Tuple[DataFrame, DataFrame]:
    """
    Import movies and rating tables.
    Select links to MovieLens datasets ("small" or "full"), and if the files don't exist, load them from the web.

    :param dataset_size: Str "small" or "full".
    :return: Tuple of two dataframes.
    """

    ## Create the folders if not exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    map_db_size_dir = {"small": DATA_SMALL_DIR, "full": DATA_FULL_DIR}
    data_dir = map_db_size_dir[dataset_size]
    movies_path = data_dir / "movies.csv"
    ratings_path = data_dir / "ratings.csv"
    suffix = "" if dataset_size == "full" else "-small"
    zip_link = f"https://files.grouplens.org/datasets/movielens/ml-latest{suffix}.zip"

    ## If the .csv files don't exist, download from the MovieLens webpage
    if not (movies_path.exists() & ratings_path.exists()):
        link_response = requests.get(zip_link)
        print(link_response)

        ## loading the temp.zip and creating a zip object
        with ZipFile(BytesIO(link_response.content), "r") as zObject:
            ## Extracting specific file in the zip into a specific location.
            zObject.extractall(path="data")

    movies_df = pd.read_csv(
        movies_path,
        usecols=["movieId", "title"],
        dtype=dict(movieId="str", title="str"),
    )
    rating_df = pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating"],
        dtype=dict(userId="str", movieId="str", rating="float32"),
    )
    return movies_df, rating_df
