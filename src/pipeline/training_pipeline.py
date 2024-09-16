import os
import sys
import argparse

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
            train_array_path,test_array_path=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
            return train_array_path,test_array_path
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_model_training(self,train_array_path,test_array_path):
        try:
            model_trainer = ModelTrainer(ModelTrainerConfig())
            model_trainer.initiate_model_trainer(train_array_path, test_array_path)
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_training_pipeline(self):
        try:
            train_data_path,test_data_path=self.start_data_ingestion()
            train_array,test_array = self.start_data_transformation(train_data_path,test_data_path)
            self.start_model_training(train_array,test_array)

        except Exception as e:
            raise CustomException(e,sys)
        
    
        
            
def main():
    parser = argparse.ArgumentParser(description="Run stages of the training pipeline.")
    parser.add_argument('stage', choices=['data_ingestion', 'data_transformation', 'model_training', 'full_pipeline'],
                        help="Specify the stage of the pipeline to run.")
    args = parser.parse_args()

    pipeline = TrainingPipeline()

    if args.stage == 'data_ingestion':
        pipeline.start_data_ingestion()

    elif args.stage == 'data_transformation':
        train_data_path = 'artifacts/train.csv'  
        test_data_path = 'artifacts/test.csv'   
        pipeline.start_data_transformation(train_data_path, test_data_path)

    elif args.stage == 'model_training':
        train_array_path = 'artifacts/train_array.npy'  # Update with actual path
        test_array_path = 'artifacts/test_array.npy'
        pipeline.start_model_training(train_array_path, test_array_path)

    elif args.stage == 'full_pipeline':
        pipeline.start_training_pipeline()


if __name__ == '__main__':
    main()