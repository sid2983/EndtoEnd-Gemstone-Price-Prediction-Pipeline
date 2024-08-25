import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.logger.loggings import logging
from src.exception.exception import CustomException
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    print('Inside evaluate_model')
    # print(models)
    # print(X_train)
    print("------------------------------>>>>>>>>>>>>>>>>>>")
    

    try:
        report = {}

        nn_model = models['neural_network']
        xgboost_model = models['xgb']
        meta_learner = models['meta_learner']

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = nn_model.fit(X_train, y_train, 
                               epochs=100, batch_size=32, 
                               validation_split=0.2, 
                               verbose=1, 
                               callbacks=[reduce_lr, early_stopping])
        
        nn_train_preds = nn_model.predict(X_train)
        nn_test_preds = nn_model.predict(X_test)

        xgboost_model.fit(X_train, y_train)

        xgboost_train_preds = xgboost_model.predict(X_train)
        xgboost_test_preds = xgboost_model.predict(X_test)

        train_meta_features = np.column_stack((xgboost_train_preds, nn_train_preds.flatten()))
        test_meta_features = np.column_stack((xgboost_test_preds, nn_test_preds.flatten()))

        meta_learner.fit(train_meta_features, y_train)

        meta_predictions = meta_learner.predict(test_meta_features)

        test_model_score = r2_score(y_test, meta_predictions)

        report['stacked_model_score'] = test_model_score

        plot_dir = os.path.join(os.getcwd(), 'artifacts', 'train_date_plots')
        os.makedirs(plot_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        loss_plot_path = os.path.join(plot_dir, f'loss_plot_{timestamp}.png')
        plt.savefig(loss_plot_path)
        plt.close()

        plt.figure()
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.show()
        mae_plot_path = os.path.join(plot_dir, f'mae_plot_{timestamp}.png')
        plt.savefig(mae_plot_path)
        plt.close()

        return report



        # for i in range(len(models)):
        #     model = list(models.values())[i]
        #     # Train model
        #     model.fit(X_train,y_train)

            

        #     # Predict Testing data
        #     y_test_pred =model.predict(X_test)

        #     # Get R2 scores for train and test data
        #     #train_model_score = r2_score(ytrain,y_train_pred)
        #     test_model_score = r2_score(y_test,y_test_pred)

        #     report[list(models.keys())[i]] =  test_model_score

        # return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

    