import json
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

from movie_recommend.constants import PKL_DIR
from movie_recommend.utils.get_recommendations import get_recommendations

# Select the size of the movie database
dataset_size = "small"
# dataset_size = "full"


if dataset_size == "small":
    knn_pkl_file = "knn_model_small.pkl"
    corr_pkl_file = "corr_model_small.pkl"
elif dataset_size == "full":
    knn_pkl_file = "knn_model.pkl"
    corr_pkl_file = "corr_model.pkl"

app = Flask(__name__)

# Load KNN model
with open(PKL_DIR / knn_pkl_file, "rb") as f:
    try:
        (
            movie_features_df__knn,
            model_knn,
            total_ratings,
            total_movie_array,
        ) = pickle.load(f)
        # print(movie_features_df__knn)
    except Exception as e:
        print(f"Error loading knn_model: {e}")
        exit()

# Load Pearson correlation model
with open(PKL_DIR / corr_pkl_file, "rb") as f:
    try:
        movie_features_df__corr, total_ratings, total_movie_array = pickle.load(f)
        # print(movie_features_df__corr)
    except Exception as e:
        print(f"Error loading corr_model: {e}")
        exit()


@app.route("/")
def home():
    """
    Renders the home page.
    """
    return render_template("home.html")


# for testing API with Postman
@app.route("/recommend_api", methods=["POST"])
def recommend_api():
    """
    Recommends movies and returns a JSON response.

    Returns:
        A JSON string containing the movie recommendations.
    """
    # Get the input data from the request JSON
    input_data = request.json["data"]

    # Check if input data is valid
    if not input_data:
        return jsonify({"message": "Invalid request"}), 400

    # Extract input data
    movie_to_compare, num_recommendations, model_type = list(input_data.values())

    # Determine which model and features to use based on the model type
    if model_type == "knn":
        movie_features_df = movie_features_df__knn
        model = model_knn
    elif model_type == "corr":
        movie_features_df = movie_features_df__corr
        model = "none"

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

    pd.set_option("display.expand_frame_repr", False)  ## Wider print-out
    print(first_line)
    print(final_table)

    return jsonify(
        {
            "json_string": json.dumps({"message": first_line}),
            "json_table": final_table.to_json(orient="table"),
        }
    )


# for HTML version
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    HTTP endpoint to get movie recommendations using Flask API.

    Returns:
        A Flask render_template object containing the header message and a Pandas DataFrame of recommended movies in HTML format.
    """
    # Extract input values from the HTML form
    input_values = request.form.values()
    movie_to_compare, num_recommendations, model_type = list(input_values)
    num_recommendations = int(num_recommendations)

    # Determine which model and features to use based on the model type
    if model_type == "knn":
        movie_features_df = movie_features_df__knn
        model = model_knn
    elif model_type == "corr":
        movie_features_df = movie_features_df__corr
        model = "none"

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

    # Return the header message and recommended movies in HTML format
    return render_template(
        "home.html", first_line=first_line, final_table=final_table.to_html()
    )


# Running the app
if __name__ == "__main__":
    app.run(debug=True)
