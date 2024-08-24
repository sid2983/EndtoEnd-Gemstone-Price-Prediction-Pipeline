import numpy as np
import pandas as pd
from src.exception.exception import CustomException
from src.logger.loggings import logging
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")



class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.ingestion_config=config

    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            
            data = pd.read_csv("data.csv")
            logging.info("Data Loaded Successfully")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Data Saved Successfully in the Artifacts Folder")

            logging.info("Performing train test split")
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("Train Test Split Completed")

            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Train and Test Data Saved Successfully in the Artifacts Folder")
            logging.info("Data Ingestion Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Data Ingestion Failed")
            raise CustomException(e,sys)



if __name__ == '__main__':
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)
    data_ingestion.initiate_data_ingestion()