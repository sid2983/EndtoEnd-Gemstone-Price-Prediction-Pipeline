import os
import sys

from src.exception.exception import CustomException
from src.logger.loggings import logging
import pandas as pd


from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.model_evaluation import ModelEvaluation, ModelEvaluationConfig


obj=DataIngestion(DataIngestionConfig())
train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation(DataTransformationConfig())
train_array,test_array=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

model_trainer = ModelTrainer(ModelTrainerConfig())
model_trainer.initiate_model_trainer(train_array, test_array)

model_evaluation = ModelEvaluation(ModelEvaluationConfig())
model_evaluation.initiate_model_evaluation(train_array, test_array)
