import pandas as pd
from pandas import DataFrame


def merged_table(movies_df: DataFrame, rating_df: DataFrame) -> DataFrame:
    """
    Merge movies and rating dfs with the new 'totalRatingCount' column.

    :param movies_df: The movies df
    :param rating_df: The rating df
    :return: Merged df with the 'totalRatingCount' column
    """

    # Combined dataframe with the column of summarized number of ratings
    df = pd.merge(movies_df, rating_df, how="left", on="movieId")
    df = df.dropna(subset=["title"])

    # Number of ratings per movie and merging with the movie df
    movie_ratingCount = (
        df.groupby(by="title")["rating"]
        .count()
        .reset_index()
        .rename(columns={"rating": "totalRatingCount"})
    )
    rating_with_totalRatingCount = df.merge(movie_ratingCount, on="title", how="left")

    return rating_with_totalRatingCount


def mean_rating_table(rating_with_totalRatingCount: DataFrame) -> DataFrame:
    """
    Calculates mean rating per movie and merges with the 'totalRatingCount' column

    :param rating_with_totalRatingCount: the movie rating per user df
    :return: mean ratings df, merged with the 'totalRatingCount' column
    """

    mean_ratings = pd.DataFrame(
        rating_with_totalRatingCount.groupby("title")["rating"].mean()
    )

    mean_ratings = mean_ratings.join(
        rating_with_totalRatingCount[["title", "totalRatingCount"]]
        .drop_duplicates()
        .set_index("title")
    )

    mean_ratings = mean_ratings.sort_values("rating", ascending=False).rename(
        columns={"rating": "mean_rating"}
    )

    return mean_ratings


def filter_movies_by_rating_count(
    rating_with_totalRatingCount: DataFrame, num_rating_threshold: int
) -> DataFrame:
    """
    Filter the movies by totalRatingCount threshold.

    :param rating_with_totalRatingCount: The rating df with the 'totalRatingCount' column
    :param num_rating_threshold: Threshold for minimum number of ratings per movie
    :return: Filtered df with the 'totalRatingCount' column
    """

    rating_movie_per_user = (
        rating_with_totalRatingCount[
            rating_with_totalRatingCount.totalRatingCount.values > num_rating_threshold
        ]
        .copy()
        .reset_index(drop=True)
    )

    return rating_movie_per_user


def pivot_title_x_users(rating_movie_per_user: DataFrame) -> DataFrame:
    """
    Rating table "movie vs users"

    :param rating_movie_per_user: output df from the 'merged_table' function

    :return: pivoted dataframe "movie vs users" - index="title", columns="userId"
    """
    rating_movie_per_user.dropna(subset=["title"], inplace=True)
    rating_movie_per_user.dropna(subset=["userId"], inplace=True)

    movie_features_df = rating_movie_per_user.pivot_table(
        index="title", columns="userId", values="rating"
    ).fillna(0)

    return movie_features_df


def pivot_users_x_title(rating_movie_per_user: DataFrame) -> DataFrame:
    """
    Rating table "users vs movie"

    :param rating_movie_per_user: output df from the 'merged_table' function

    :return: pivoted dataframe "users vs movie" - index="userId", columns="title"
    """
    rating_movie_per_user.dropna(subset=["title"], inplace=True)
    rating_movie_per_user.dropna(subset=["userId"], inplace=True)

    movie_features_df = rating_movie_per_user.pivot_table(
        index="userId", columns="title", values="rating"
    )

    return movie_features_df
