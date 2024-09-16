import numpy as np
import pandas as pd
from src.logger.loggings import logging
from src.exception.exception import CustomException

import os
import sys

from dataclasses import dataclass
from pathlib import Path
from src.utils.utils import save_object, load_object, evaluate_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, RidgeCV, LassoCV, ElasticNetCV, SGDRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor,XGBRFRegressor
import xgboost as xgb

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, AdamW, RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau


# just to check 
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')



class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.model_trainer_config=config

    
    def initiate_model_trainer(self,train_array_path, test_array_path):
        try:
            logging.info("Model Training Started")
            logging.info("Splitting Dependent and Independent variables from train and test data")


            train_array = np.load(train_array_path)
            test_array = np.load(test_array_path)

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            ## Using ANN & XGBRegressor Stacking Model

            nn_model = Sequential([
                Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.0001)),
                Dense(256, activation='relu', kernel_regularizer=l2(0.0001)),
                Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),
                Dense(1)  # Output layer for regression
            ])

            nn_model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.0001),
                 loss='mean_squared_error',
                 metrics=['mae'])
            
            

            ## XGBRegressor Model
            xgboost_model = xgb.XGBRegressor(learning_rate=0.01, n_estimators=1000, max_depth=5, alpha=10, gamma=0.1)


            ## Stacking Model
            
            meta_learner = Ridge()

            # send dictionary of models to evaluate_model function
            models = {
                'neural_network': nn_model,
                'xgb': xgboost_model,
                'meta_learner': meta_learner
            }

            logging.info("Model Training Initiated")
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test,models)
            print(model_report)

            logging.info(f'Model Report : {model_report}')

            logging.info("Model Training Completed")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=models
            )

            logging.info("Model Training Completed")


        except Exception as e:
            logging.info("Model Training Failed")
            raise CustomException(e,sys)


