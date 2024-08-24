import numpy as np
import pandas as pd
from src.logger.loggings import logging
from src.exception.exception import CustomException

import os
import sys
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from pathlib import Path
from src.utils.utils import save_object, load_object


@dataclass
class DataTransformationConfig:
    pass



class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config=config

    
    def initiate_data_transformation(self):
        try:
            logging.info("Data Transformation Started")

            logging.info("Data Transformation Completed")
        except Exception as e:
            logging.info("Data Transformation Failed")
            raise CustomException(e,sys)

