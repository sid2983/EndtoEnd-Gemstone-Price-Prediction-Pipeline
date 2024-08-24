import numpy as np
import pandas as pd
from src.logger.loggings import logging
from src.exception.exception import CustomException

import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    pass



class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.config=config

    
    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion Started")

            logging.info("Data Ingestion Completed")
        except Exception as e:
            logging.info("Data Ingestion Failed")
            raise CustomException(e,sys)

