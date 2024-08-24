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


@dataclass
class ModelTrainerConfig:
    pass



class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config=config

    
    def initiate_model_trainer(self):
        try:
            logging.info("Model Training Started")
            self.read_data()
            self.split_data()
            self.scale_data()
            logging.info("Model Training Completed")
        except Exception as e:
            logging.info("Model Training Failed")
            raise CustomException(e,sys)

