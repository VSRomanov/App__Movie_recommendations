from pandas import DataFrame

from movie_recommend.utils.recommendation_algorithms import recommendation_corr, recommendation_knn

class ModelType:
    """The ModelType interface declares the fields and operations that all specific model types must implement."""
    string = ""


class ModelTypeKnn(ModelType):
    """KNN model type class."""
    string = "knn"

    def get_recommendations(self, features_df, model, movie_to_compare, n_recommend, total_ratings):
        return recommendation_knn(features_df, model, movie_to_compare, n_recommend, total_ratings)

    def get_movie_array(self, df: DataFrame):
        return df.index


class ModelTypeCorr(ModelType):
    """Pearson correlation model type class."""
    string = "corr"

    def get_recommendations(self, features_df, model, movie_to_compare, n_recommend, total_ratings):
        return recommendation_corr(features_df, movie_to_compare, n_recommend, total_ratings)

    def get_movie_array(self, df: DataFrame):
        return df.columns


def get_model_type_class_by_name(model_type: str) -> ModelType:
    """Get model type class by name."""
    model_types = {
        "knn": ModelTypeKnn,
        "corr": ModelTypeCorr,
    }
    return model_types.get(model_type, ModelType)
