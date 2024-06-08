"""
This script visualizes the relationship between the 'mean_rating' and 'totalRatingCount' for movies.
It generates PNG files and saves them in the OUTPUT_FIG folder.

Users can choose the minimum mean rating and minimum number of ratings per movie to consider,
as well as the size of the dataset (either "small" or "full").
"""
import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

import movie_recommend.constants as c
from movie_recommend.utils.get_databases import get_db
from movie_recommend.utils.table_formatting import mean_rating_table, merged_table, filter_movies_by_rating_count

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def data_visualisation(total_ratings: DataFrame, dataset_size: str, num_rating_threshold: int,
    rating_threshold: float, folder_path: str
) -> None:
    """Generate a joint plot to visualize the relationship between movie ratings and the total rating count
    for movies with a minimum mean rating and minimum number of ratings."""

    # Filter movies with more than the threshold number of ratings and more than the mean rating threshold
    filtered_movies = filter_movies_by_rating_count(total_ratings, num_rating_threshold)
    filtered_movies = filtered_movies[filtered_movies["mean_rating"] >= rating_threshold]

    logging.info("Top 20 movies after filtering: %s", filtered_movies.head(20))

    # Create output folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # Create joint plot and save to file
    sns.set_style("white")
    sns.jointplot(
        x="mean_rating", y="totalRatingCount", data=filtered_movies, alpha=0.5
    )
    plt.tight_layout()

    output_file_path = os.path.join(folder_path, f"rating_vs_totalRatingCount_{dataset_size}.png")
    plt.savefig(output_file_path)
    logging.info("Visualization saved to: %s", output_file_path)


def main(dataset_size: str, rating_threshold: int, min_rating: float) -> None:
    """Main function to load data, compute mean ratings, and generate visualizations."""
    try:
        # Import movies and ratings dataframes
        movies_df, rating_df = get_db(dataset_size)

        logging.info("The total number of movies in the database: %d", len(movies_df.index))
        logging.info("The total number of ratings in the database: %d", len(rating_df.index))

        movie_rating_df = merged_table(movies_df, rating_df)

        # Compute mean ratings for each movie
        all_ratings = mean_rating_table(movie_rating_df)

        data_visualisation(all_ratings, dataset_size, rating_threshold, min_rating, c.OUTPUT_FIG)
    except Exception as e:
        logging.error("An error occurred: %s", e)


if __name__ == "__main__":
    # Configuration
    dataset_size = "small"; rating_threshold = 10
    # dataset_size = "full"; rating_threshold = 500

    # Minimum mean rating to consider
    min_rating = 0
    # min_rating = 3.5

    main(dataset_size, rating_threshold, min_rating)