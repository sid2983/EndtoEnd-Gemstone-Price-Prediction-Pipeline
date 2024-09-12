import os
import sys

from src.exception.exception import CustomException
from src.logger.loggings import logging
import pandas as pd


from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.model_evaluation import ModelEvaluation, ModelEvaluationConfig






class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion=DataIngestion(DataIngestionConfig())
            train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
            return train_data_path,test_data_path   
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_transformation(self,train_data_path,test_data_path):
        try:
            data_transformation=DataTransformation(DataTransformationConfig())
            train_array,test_array=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
            return train_array,test_array
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_model_training(self,train_array,test_array):
        try:
            model_trainer = ModelTrainer(ModelTrainerConfig())
            model_trainer.initiate_model_trainer(train_array, test_array)
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_training_pipeline(self):
        try:
            train_data_path,test_data_path=self.start_data_ingestion()
            train_array,test_array = self.start_data_transformation(train_data_path,test_data_path)
            self.start_model_training(train_array,test_array)

        except Exception as e:
            raise CustomException(e,sys)
        
    
        
            
