import numpy as np
import pandas as pd
from src.logger.loggings import logging
from src.exception.exception import CustomException

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class ModelEvaluationConfig:
    pass



class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config=config

    
    def initiate_model_evaluation(self):
        try:
            logging.info("Model Evaluation Started")
            self.read_data()
            self.split_data()
            self.scale_data()
            logging.info("Model Evaluation Completed")
        except Exception as e:
            logging.info("Model Evaluation Failed")
            raise CustomException(e,sys)

