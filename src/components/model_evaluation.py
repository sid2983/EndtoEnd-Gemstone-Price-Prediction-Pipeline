import numpy as np
import pandas as pd
from src.logger.loggings import logging
from src.exception.exception import CustomException

from src.utils.utils import save_object, load_object, evaluate_model

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
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.model_evaluation_config=config
        logging.info("Model Evaluation Started")
        

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        
        logging.info(f"Evaluation Metrics Captured : RMSE : {rmse}, MAE : {mae}, R2 : {r2}")
        return rmse, mae, r2
    
    def initiate_model_evaluation(self,train_array,test_array):

        try:
            logging.info("Model Evaluation Started")
            X_test,y_test=(test_array[:,:-1], test_array[:,-1])
            # model_path=os.path.join("artifacts","model.pkl")
            # model=load_object(model_path)

            # y_pred=model.predict(X_test)

            loaded_models = load_object(self.model_evaluation_config.trained_model_file_path)
            logging.info(f"Loaded Models : {loaded_models}")

            nn_predictions = loaded_models['neural_network'].predict(X_test)
            print("Neural Network Predictions:", nn_predictions[:10])

            xgb_predictions = loaded_models['xgb'].predict(X_test)
            print("XGBoost Predictions:", xgb_predictions[:10])

            
            meta_test_features = np.column_stack((
                loaded_models['xgb'].predict(X_test),
                loaded_models['neural_network'].predict(X_test).flatten()
            ))
            meta_predictions = loaded_models['meta_learner'].predict(meta_test_features)
            print("Meta-Learner Predictions:", meta_predictions[:10])

            rmse, mae, r2=self.eval_metrics(y_test,meta_predictions)
            logging.info(f"Model Evaluation Completed with metrics: RMSE : {rmse}, MAE : {mae}, R2 : {r2}")
            logging.info("Model Evaluation Completed")


        except Exception as e:
            logging.info("Model Evaluation Failed")
            raise CustomException(e,sys)

