"""
Visualization of 'mean_rating' vs 'totalRatingCount' per movie: png files in the OUTPUT_FIG folder.

The script allows the user to choose the minimum mean rating and minimum number of ratings per movie to consider,
as well as the size of the dataset (either "small" or "full").
"""

from movie_recommend.constants import OUTPUT_FIG
from movie_recommend.utils.data_visualisation import ratings_visualisation
from movie_recommend.utils.get_databases import get_db
from movie_recommend.utils.table_formatting import mean_rating_table, merged_table

if __name__ == "__main__":
    ## Small or full dataset?
    # dataset_size = "small"
    dataset_size = "full"

    ## Minimum number of ratings per movie
    # num_rating_threshold = 10
    num_rating_threshold = 500

    ## Minimum mean rating to consider
    rating_threshold = 0
    # rating_threshold = 3.5

    # Import movies and ratings dataframes
    movies_df, rating_df = get_db(dataset_size)

    # Print information about tables
    print("\nThe total number of movies in the database:", len(movies_df.index))
    print("The total number of ratings in the database:", len(rating_df.index))

    # Merge dataframes
    rating_with_totalRatingCount = merged_table(movies_df, rating_df)

    # Compute mean ratings for each movie
    total_ratings = mean_rating_table(rating_with_totalRatingCount)

    ratings_visualisation(
        total_ratings, dataset_size, num_rating_threshold, rating_threshold, OUTPUT_FIG
    )
