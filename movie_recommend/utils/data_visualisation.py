import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from movie_recommend.utils.table_formatting import filter_movies_by_rating_count


def ratings_visualisation(
        total_ratings: DataFrame,
        dataset_size: str,
        num_rating_threshold: int,
        rating_threshold: float,
        folder_path: str,
) -> None:
    """
    Generate a joint plot to visualize the relationship between movie ratings and the total rating count
    for movies with a minimum mean rating and minimum number of ratings.

    :param total_ratings: The DataFrame containing movie ratings data
    :param dataset_size: The size of the dataset to retrieve, either 'small' or 'large'
    :param num_rating_threshold: The minimum number of ratings required for a movie to be included in the plot
    :param rating_threshold: The minimum mean rating required for a movie to be included in the plot
    :param folder_path: The path to the folder where the plot image will be saved
    :return: None
    """

    # Filter movies with more than the threshold number of ratings and more than the mean rating threshold
    rating_movie_per_user = filter_movies_by_rating_count(
        total_ratings, num_rating_threshold
    )
    rating_movie_per_user = rating_movie_per_user[
        rating_movie_per_user["mean_rating"] >= rating_threshold
        ]
    # print(rating_movie_per_user.head(20))

    # Create output folder if it does not exist
    folder_path.mkdir(parents=True, exist_ok=True)

    # Create joint plot and save to file
    sns.set_style("white")
    sns.jointplot(
        x="mean_rating", y="totalRatingCount", data=rating_movie_per_user, alpha=0.5
    )
    plt.tight_layout()
    plt.savefig(folder_path / f"rating_vs_totalRatingCount_{dataset_size}.png")

    print("\nDone!")
