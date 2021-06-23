import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import pickle
import pandas as pd

labels = ["setosa", "versicolor", "virginica"]


class PythonPredictor:
    def __init__(self, config):

        self.deployment_name = config["deployment_name"]
        with open(config["model_path"], "rb") as inp:
            self.model = pickle.load(inp)

    def get_model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self.model is None:
            with open("model_bg_A.pkl", "rb") as inp:
                self.model = pickle.load(inp)
        return self.model

    def predict(self, payload):

        member_id = payload["member_id"]

        clf = self.get_model()

        # Load user data // put mocked data
        user_data = pd.DataFrame(
            {
                "b_g": [
                    "290 0",
                    "626 1",
                    "259 1",
                    "2 1",
                    "485 1",
                    "425 1",
                    "471 1",
                    "14 1",
                ],
                "total_hits": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

        if user_data.empty:
            print("Member ID {} has no user interactions".format(member_id))
            return []

        raw_predictions = clf.predict(user_data)

        if raw_predictions is None or len(raw_predictions.brand.keys()) < 1:
            print("Member ID {} has no user predictions".format(member_id))
            return []

        predictions = list(
            map(
                lambda k: Prediction(
                    brand_id=raw_predictions.brand[k],
                    gender=raw_predictions.gender[k],
                    score=raw_predictions.score[k],
                    liked=raw_predictions.liked[k],
                ),
                raw_predictions.brand.keys(),
            )
        )

        predictions_dict = [prediction.__dict__ for prediction in predictions]

        return {
            "deployment_name": self.deployment_name,
            "predictor_info": "ROLLING UPDATES DEMO",
            "predictions": predictions_dict,
        }


class Prediction:
    brand_id: int
    score: float
    liked: bool
    gender: int

    def __init__(self, brand_id: int, score: float, liked: bool, gender: int):
        self.brand_id = brand_id
        self.score = score
        self.liked = liked
        self.gender = gender
