import time
import logging
import pandas as pd
from pandas import DataFrame

import movie_recommend.constants as c

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def merged_table(movies_df: DataFrame, rating_df: DataFrame) -> DataFrame:
    """Merge movies and ratings dataframes, adding a 'totalRatingCount' column."""
    logging.info("Counting number of ratings per movie")
    df = pd.merge(movies_df, rating_df, how="left", on=c.MOVIE_ID).dropna(subset=[c.TITLE])

    # Number of ratings per movie
    movie_rating_count = (
        df.groupby(c.TITLE)[c.RATING]
        .count()
        .reset_index()
        .rename(columns={c.RATING: c.TOTAL_RATING_COUNT})
    )
    result_df = df.merge(movie_rating_count, on=c.TITLE, how="left")

    return result_df


def mean_rating_table(movie_rating_df: DataFrame) -> DataFrame:
    """Calculates mean rating per movie and merges with the 'totalRatingCount' column"""
    logging.info("Calculating mean rating per movie")
    mean_ratings = movie_rating_df.groupby(c.TITLE)[c.RATING].mean()

    mean_ratings_df = mean_ratings.to_frame().join(
        movie_rating_df[[c.TITLE, c.TOTAL_RATING_COUNT]]
        .drop_duplicates()
        .set_index(c.TITLE)
    )

    mean_ratings_df = mean_ratings_df.sort_values(c.RATING, ascending=False).rename(
        columns={c.RATING: c.MEAN_RATING}
    )

    return mean_ratings_df.reset_index()


def filter_movies_by_rating_count(movie_rating_df: DataFrame, rating_threshold: int) -> DataFrame:
    """Filter movies by a minimum number of ratings."""
    logging.info("Filtering movies by rating count")
    filtered_df = movie_rating_df[movie_rating_df.totalRatingCount.values > rating_threshold
                                  ].copy().reset_index(drop=True)
    return filtered_df


def pivot_ratings(rating_movie_per_user: DataFrame, index: str, columns: str) -> DataFrame:
    """Create a pivot table of movie ratings."""
    start_time = time.time()

    logging.info("Creating a pivot table of movie ratings...")
    rating_movie_per_user.dropna(subset=[index, columns], inplace=True)
    pivot_df = rating_movie_per_user.pivot_table(index=index, columns=columns, values=c.RATING)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Done! Time taken to create pivot table: %.2f seconds", elapsed_time)

    return pivot_df
