import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import mlflow
import mlflow.keras
import mlflow.xgboost
import mlflow.sklearn
from mlflow import log_params, log_metrics, log_artifact, log_figure
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
        

        plot_dir = os.path.join('artifacts', 'train_date_plots')
        os.makedirs(plot_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        report = {}

        nn_model = models['neural_network']
        xgboost_model = models['xgb']
        meta_learner = models['meta_learner']

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        with mlflow.start_run(run_name="Neural Network Training") as run:

            history = nn_model.fit(X_train, y_train, 
                                epochs=100, batch_size=32, 
                                validation_split=0.2, 
                                verbose=1, 
                                callbacks=[reduce_lr, early_stopping])
            

            for epoch in range(len(history.history['loss'])):
                mlflow.log_metric('train_loss', history.history['loss'][epoch], step=epoch)
                mlflow.log_metric('val_loss', history.history['val_loss'][epoch], step=epoch)
                mlflow.log_metric('train_mae', history.history['mae'][epoch], step=epoch)
                mlflow.log_metric('val_mae', history.history['val_mae'][epoch], step=epoch)

            nn_params = {
                'input_dim': nn_model.input_shape[1],
                'epochs': 100,
                'batch_size': 32,
                'optimizer': 'AdamW',
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'kernel_regularizer': 'l2(0.0001)'
            }
            mlflow.log_params(nn_params)

            mlflow.keras.log_model(nn_model, "neural_network_model",registered_model_name="NeuralNetworkModel")
        
            nn_train_preds = nn_model.predict(X_train)
            nn_test_preds = nn_model.predict(X_test)

            mlflow.log_metric('nn_train_r2_score', r2_score(y_train, nn_train_preds))
            mlflow.log_metric('nn_test_r2_score', r2_score(y_test, nn_test_preds))

            

            plt.figure()
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss vs Epoch')
            plt.tight_layout()
            # mlflow.log_figure(plt.gcf(), 'loss_plot.png')
            loss_plot_path = os.path.join(plot_dir, f'loss_plot_{timestamp}.png')
            plt.savefig(loss_plot_path) 
            plt.close()
            mlflow.log_artifact(loss_plot_path)

            plt.figure()
            plt.plot(history.history['mae'], label='Train MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.title('MAE vs Epoch')
            plt.tight_layout()
            # mlflow.log_figure(plt.gcf(), 'mae_plot.png')
            mae_plot_path = os.path.join(plot_dir, f'mae_plot_{timestamp}.png')
            plt.savefig(mae_plot_path)
            plt.close()
            mlflow.log_artifact(mae_plot_path)
            logging.info('Plots saved successfully')
            logging.info('NN Model training completed  and MLflow logs saved successfully')

        with mlflow.start_run(run_name="XGBoost Training") as run:

            xgboost_model.fit(X_train, y_train)
            mlflow.xgboost.log_model(xgboost_model, "xgboost_model",registered_model_name="XGBoostModel")

            xgb_params = xgboost_model.get_params()
            mlflow.log_params(xgb_params)

            xgboost_train_preds = xgboost_model.predict(X_train)
            xgboost_test_preds = xgboost_model.predict(X_test)

            mlflow.log_metric('xgboost_train_r2_score', r2_score(y_train, xgboost_train_preds))
            mlflow.log_metric('xgboost_test_r2_score', r2_score(y_test, xgboost_test_preds))

            logging.info('XGBoost Model training completed and MLflow logs saved successfully')


        with mlflow.start_run(run_name="Stacking Model Training") as run:

            train_meta_features = np.column_stack((xgboost_train_preds, nn_train_preds.flatten()))
            test_meta_features = np.column_stack((xgboost_test_preds, nn_test_preds.flatten()))

            meta_learner.fit(train_meta_features, y_train)
            meta_predictions = meta_learner.predict(test_meta_features)

            test_model_score = r2_score(y_test, meta_predictions)
            test_model_mae = mean_absolute_error(y_test, meta_predictions)
            test_model_mse = mean_squared_error(y_test, meta_predictions)
            test_model_rmse = np.sqrt(test_model_mse)

            mlflow.log_metric('stacked_model_r2_score', test_model_score)
            mlflow.log_metric('stacked_model_mae', test_model_mae)
            mlflow.log_metric('stacked_model_rmse', test_model_rmse)
            mlflow.log_params({'meta_learner': 'Ridge'})
            mlflow.sklearn.log_model(meta_learner, "meta_learner_model",registered_model_name="StackingModel")

        
            logging.info('Stacking Model training completed and MLflow logs saved successfully')
        
        report['stacked_model_score'] = test_model_score





        return report




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

    