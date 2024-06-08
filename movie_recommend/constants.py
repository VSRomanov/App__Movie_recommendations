import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(REPO_DIR, "raw_data")
DATA_FULL_DIR = os.path.join(DATA_DIR, "ml-latest")
DATA_SMALL_DIR = os.path.join(DATA_DIR, "ml-latest-small")

MOVIES_CSV = "movies.csv"
RATINGS_CSV = "ratings.csv"
ZIP_LINK_FULL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
ZIP_LINK_SMALL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
LAST_ETAG_FILE = "last_etag.txt"
LAST_MODIFIED_FILE = "last_modified.txt"

PKL_DIR = os.path.join(REPO_DIR, "app_data")

OUTPUT_DIR = os.path.join(REPO_DIR, "output")
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "figures")

# column names
MOVIE_ID = "movieId"
USER_ID = "userId"
TITLE = "title"
RATING = "rating"
TOTAL_RATING_COUNT = "totalRatingCount"
MEAN_RATING = "mean_rating"