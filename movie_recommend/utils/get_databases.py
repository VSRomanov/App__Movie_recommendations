import os
from typing import Tuple
from zipfile import ZipFile

import requests
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
import logging

import movie_recommend.constants as c

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_headers(url: str) -> dict:
    """Fetch headers for the given URL."""
    response = requests.head(url)
    response.raise_for_status()
    return response.headers


def load_last_values(dir_path: str, etag_file: str, modified_file: str) -> Tuple[str, str]:
    """Load last ETag and Last-Modified values from files."""
    last_etag = None
    last_modified = None
    if os.path.exists(os.path.join(dir_path, etag_file)):
        with open(os.path.join(dir_path, etag_file), 'r') as f:
            last_etag = f.read().strip()
    if os.path.exists(os.path.join(dir_path, modified_file)):
        with open(os.path.join(dir_path, modified_file), 'r') as f:
            last_modified = f.read().strip()
    return last_etag, last_modified


def save_current_values(etag_file: str, modified_file: str, etag: str, modified: str) -> None:
    """Save current ETag and Last-Modified values to files."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(etag_file), exist_ok=True)
    os.makedirs(os.path.dirname(modified_file), exist_ok=True)

    # Write the ETag and Last-Modified values to their respective files
    with open(etag_file, 'w') as f:
        f.write(etag)
    with open(modified_file, 'w') as f:
        f.write(modified)


def check_file_change(dir_path: str, url: str, etag_file: str, modified_file: str, overwrite=True) -> bool:
    """Check if the file has changed."""
    headers = fetch_headers(url)
    etag = headers.get('ETag')
    last_modified = headers.get('Last-Modified')

    last_etag, last_modified_saved = load_last_values(dir_path, etag_file, modified_file)
    file_changed = (etag != last_etag) or (last_modified != last_modified_saved)

    if file_changed and overwrite:
        save_current_values(os.path.join(dir_path, etag_file), os.path.join(dir_path, modified_file), etag,
                            last_modified)
    return file_changed


def download_and_extract_zip(zip_link: str, dir_path: str) -> None:
    """Download and extract a zip file."""
    temp_zip_path = "temp.zip"

    # Download the zip file in chunks
    response = requests.get(zip_link, stream=True)
    response.raise_for_status()  # Raise an exception if the request was unsuccessful

    # For visualization the progress bar
    total_size = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

    try:
        with open(temp_zip_path, 'wb') as temp_zip_file:
            # Iterate over the response content in chunks of 8192 bytes (8 KB)
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_zip_file.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()
    except Exception as e:
        logging.error("Error downloading data: %s", e)
        exit(1)
    logging.info("Zip file downloaded successfully")

    # Create a zip object and extract
    try:
        with ZipFile(temp_zip_path, "r") as zip_file:
            zip_file.extractall(path=c.DATA_DIR)
    except Exception as e:
        logging.error("Error unzipping data: %s", e)
        exit(1)
    logging.info("CSV files extracted successfully")

    # Delete the temporary zip file
    os.remove(temp_zip_path)


def set_folders_files(dataset_size: str) -> Tuple[str, str, str, str, str, str]:
    """Set folder paths and file links based on dataset size."""
    os.makedirs(c.DATA_DIR, exist_ok=True)
    map_db_size_dir = {"small": c.DATA_SMALL_DIR, "full": c.DATA_FULL_DIR}

    dir_path = map_db_size_dir[dataset_size]
    movies_path = os.path.join(dir_path, c.MOVIES_CSV)
    ratings_path = os.path.join(dir_path, c.RATINGS_CSV)
    zip_link = c.ZIP_LINK_FULL if dataset_size == "full" else c.ZIP_LINK_SMALL
    last_etag_file = os.path.join(dir_path, c.LAST_ETAG_FILE)
    last_modified_file = os.path.join(dir_path, c.LAST_MODIFIED_FILE)

    return dir_path, movies_path, ratings_path, zip_link, last_etag_file, last_modified_file


def get_db(dataset_size: str) -> Tuple[DataFrame, DataFrame]:
    """Import movies and rating tables.
    Select links to MovieLens datasets ("small" or "full"), and if the files don't exist, load them from the webpage.
    """

    dir_path, movies_path, ratings_path, zip_link, last_etag_file, last_modified_file = set_folders_files(dataset_size)

    # Check if zip file was modified at the source page
    file_changed = check_file_change(dir_path, zip_link, last_etag_file, last_modified_file)

    # If the db files don't exist or were modified in the MovieLens webpage, download them
    if file_changed or not (os.path.exists(movies_path) and os.path.exists(ratings_path)):
        logging.info("The database has to be updated. Downloading the new file...")
        download_and_extract_zip(zip_link, dir_path)
    else:
        logging.info("The database has not changed. No need to download from the web")

    movies_df = pd.read_csv(movies_path, usecols=["movieId", "title"], dtype=dict(movieId="str", title="str"))
    rating_df = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"],
                            dtype=dict(userId="str", movieId="str", rating="float32"))

    return movies_df, rating_df


def main(dataset_size: str = "full") -> None:
    """For local testing"""
    dir_path, movies_path, ratings_path, zip_link, last_etag_file, last_modified_file = set_folders_files(dataset_size)
    file_changed = check_file_change(dir_path, zip_link, last_etag_file, last_modified_file, overwrite = False)

    if file_changed:
        logging.info("The file has changed. You can download the new file")
    else:
        logging.info("The file has not changed. No need to download")


if __name__ == "__main__":
    # dataset_size = "small"
    dataset_size = "full"
    main(dataset_size)
