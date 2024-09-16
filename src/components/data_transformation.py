import numpy as np
import pandas as pd
from src.logger.loggings import logging
from src.exception.exception import CustomException

import os
import sys
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from pathlib import Path
from src.utils.utils import save_object, load_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')



class VolumeSurfaceAreaTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['volume'] = X['x'] * X['y'] * X['z']
        X['surface_area'] = 2 * (X['x'] * X['y'] + X['x'] * X['z'] + X['y'] * X['z'])
        X.drop(['x', 'y', 'z'], axis=1, inplace=True)
        
        return X

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.data_transformation_config=config

    def get_data_transformation_pipeline(self):
        try:
            logging.info("Data Transformation Pipeline Started")
            numeric_features = ['carat', 'depth', 'table', 'surface_area', 'volume']
            categorical_features = ['cut', 'color', 'clarity']

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']


            logging.info("Data Transformation Pipeline Initiated")

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('std_scaler', StandardScaler()),
            ])

            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('std_scaler', StandardScaler()),
            ])

        
            full_pipeline = Pipeline([
                ('volumesurfacearea', VolumeSurfaceAreaTransformer()),
                ('preprocessor', ColumnTransformer([
                    ('num', num_pipeline, numeric_features),
                    ('cat', cat_pipeline, categorical_features),
                ]))
            ])

            logging.info("Data Transformation Pipeline Completed")
            return full_pipeline

            
            # return preprocessor
        except Exception as e:
            logging.info("Data Transformation Pipeline Failed. Exception Occured")
            raise CustomException(e,sys)

    
    def initiate_data_transformation(self,train_path:str,test_path:str):
        try:
            logging.info("Data Transformation Started")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            preprocessing_obj = self.get_data_transformation_pipeline()

            

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            logging.info("Data Transformation Completed")
            #data head after transformation with column names in df format
            logging.info(f'Train Data Head after Transformation : \n{pd.DataFrame(input_feature_train_arr,columns=['carat','cut','color','clarity','depth','table','volume','surface_area']).head().to_string()}')
            logging.info(f'Test Data Head after Transformation : \n{pd.DataFrame(input_feature_test_arr,columns=['carat','cut','color','clarity','depth','table','volume','surface_area']).head().to_string()}')


            # train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            # test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            np.save('artifacts/train_array.npy', np.c_[input_feature_train_arr, np.array(target_feature_train_df)])
            np.save('artifacts/test_array.npy', np.c_[input_feature_test_arr, np.array(target_feature_test_df)])

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved successfully")
            logging.info("Data Transformation Completed")

            return 'artifacts/train_array.npy', 'artifacts/test_array.npy'









        except Exception as e:
            logging.info("Data Transformation Failed")
            raise CustomException(e,sys)

