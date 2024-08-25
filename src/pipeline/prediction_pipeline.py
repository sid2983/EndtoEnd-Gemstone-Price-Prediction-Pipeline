import os
import sys
import numpy as np
from src.logger.loggings import logging
from src.exception.exception import CustomException
import pandas as pd
from src.utils.utils import save_object, load_object



class PredictPipeline:

    def __init__(self):
        print("Prediction Pipeline initiated")

    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            loaded_models = load_object(model_path)

            preprocessor=load_object(preprocessor_path)
            scaled_fea=preprocessor.transform(features)


            meta_test_features = np.column_stack((
                loaded_models['xgb'].predict(scaled_fea),
                loaded_models['neural_network'].predict(scaled_fea).flatten()
            ))

            meta_predictions = loaded_models['meta_learner'].predict(meta_test_features)
            print("Meta-Learner Predictions:", meta_predictions[:10])

            return meta_predictions





        except Exception as e:
            logging.info("Prediction Pipeline Failed")
            raise CustomException(e,sys)
        


class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity


        



    def get_data_as_df(self):

        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
                } 
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            logging.info("Dataframe Creation Failed")
            logging.info("Prediction Pipeline Failed")
            raise CustomException(e,sys)
