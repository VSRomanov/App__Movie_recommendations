import json

import pandas as pd
from flask import Flask, jsonify, render_template, request

from movie_recommend.movie_recommendations import MovieRecommend

# Select the size of the movie database
#dataset_size = "small"
dataset_size = "full"

app = Flask(__name__)


@app.route("/")
def home():
    """Renders the home page."""
    return render_template("home.html")


# for testing API with Postman
@app.route("/recommend_api", methods=["POST"])
def recommend_api():
    """Recommends movies and returns a JSON response."""

    input_data = request.json["data"]

    if not input_data:
        return jsonify({"message": "Invalid request"}), 400

    movie_to_compare, n_recommend, model_type = list(input_data.values())
    n_recommend = int(n_recommend)

    # Create an instance of MovieRecommend and get the movie recommendations
    first_line, final_table = MovieRecommend(model_type=model_type, db_size=dataset_size, n_recommend=n_recommend).launch(movie_to_compare)

    # Set pandas options for wider print-out
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)

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
    """HTTP endpoint to get movie recommendations using Flask API."""
    # Extract input values from the HTML form
    input_values = request.form.values()
    movie_to_compare, n_recommend, model_type = list(input_values)
    n_recommend = int(n_recommend)

    # Create an instance of MovieRecommend and get the movie recommendations
    first_line, final_table = MovieRecommend(model_type=model_type, db_size=dataset_size, n_recommend=n_recommend).launch(movie_to_compare)

    # Return the header message and recommended movies in HTML format
    return render_template("home.html", first_line=first_line, final_table=final_table.to_html())


# Running the app
if __name__ == "__main__":
    app.run(debug=True)
