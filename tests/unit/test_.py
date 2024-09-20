import os
import pickle
from src.pipeline.prediction_pipeline import PredictPipeline,CustomData
import numpy as np
import pandas as pd

# Adding a dummy test to check the CI_CD
def dummy_test():
    assert 1==1



def get_artifacts_path(filename):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(project_root, 'artifacts', filename)

# print(get_artifacts_path('model.pkl'))

def test_data_ingestion():
    # Assuming the data ingestion step creates 'train.csv' and 'test.csv' in artifacts folder
    assert os.path.exists(get_artifacts_path('raw.csv')), "Raw data not found!"
    assert os.path.exists(get_artifacts_path('train.csv')), "Train data not found!"
    assert os.path.exists(get_artifacts_path('test.csv')), "Test data not found!"
    assert os.path.getsize(get_artifacts_path('train.csv')) > 0, "Train data is empty!"
    assert os.path.getsize(get_artifacts_path('test.csv')) > 0, "Test data is empty!"



def test_data_transformation():
    # Test for presence of preprocessor and transformed arrays
    assert os.path.exists(get_artifacts_path('preprocessor.pkl')), "Preprocessor file not found!"
    # assert os.path.exists(get_artifacts_path('train_array.npy')), "Train array not found!"
    # assert os.path.exists(get_artifacts_path('test_array.npy')), "Test array not found!"
    
    # # Optional: Check if arrays have data
    # train_array = np.load(get_artifacts_path('train_array.npy'))
    # test_array = np.load(get_artifacts_path('test_array.npy'))
    
    # assert train_array.shape[0] > 0, "Train array is empty!"
    # assert test_array.shape[0] > 0, "Test array is empty!"




def test_model_training():
    # Test for the existence of the model file
    assert os.path.exists(get_artifacts_path('model.pkl')), "Model file not found!"
    
    # Check if the model can be loaded
    with open(get_artifacts_path('model.pkl'), 'rb') as model_file:
        model = pickle.load(model_file)
    
    assert model is not None, "Failed to load the model!"






# # Predictions and model evaluation


# Load the trained model
def load_model():
    with open(get_artifacts_path('model.pkl'), 'rb') as model_file:
        return pickle.load(model_file)
    

def test_single_prediction():
    model = load_model()
    
    # Define a sample input
    sample_input = [[0.33,"Premium","E","VS2",61.7,59.0,4.39,4.43,2.72]]  
    data=CustomData(
            carat=float(sample_input[0][0]),
            depth=float(sample_input[0][4]),
            table=float(sample_input[0][5]),
            x=float(sample_input[0][6]),
            y=float(sample_input[0][7]),
            z=float(sample_input[0][8]),
            cut=sample_input[0][1],
            color=sample_input[0][2],
            clarity=sample_input[0][3]
        )
    final_data=data.get_data_as_df()
    predict_pipeline=PredictPipeline()

    pred=predict_pipeline.predict(final_data)

    result=round(pred[0],2)
    
    
    
    assert result is not None, "Prediction for single input failed!"

# print(test_single_prediction())


def load_test_data(file_path):
    # Load the test data, excluding first (ID) and last (price) columns
    df = pd.read_csv(file_path)
    return df.iloc[2:50, 1:-1]  # Exclude first and last columns


def test_batch_prediction():
    test_file = get_artifacts_path('test.csv')
    assert os.path.exists(test_file), "Test data file not found!"
    
    # Load the batch data excluding ID and price columns
    batch_data = load_test_data(test_file)
    
    # Check if the batch_data is not empty
    assert not batch_data.empty, "Batch data is empty!"

    predict_pipeline = PredictPipeline()

    # Prepare and collect predictions for the entire batch
    predictions = []

    for i in range(len(batch_data)):
        # Extract row data
        row = batch_data.iloc[i].tolist()

        # Use CustomData class to align the data correctly
        data = CustomData(
            carat=float(row[0]),
            depth=float(row[4]),
            table=float(row[5]),
            x=float(row[6]),
            y=float(row[7]),
            z=float(row[8]),
            cut=row[1],
            color=row[2],
            clarity=row[3]
        )

        # Convert the row to a DataFrame format expected by the model
        final_data = data.get_data_as_df()
        
        # Perform the prediction
        pred = predict_pipeline.predict(final_data)
        
        # Append the prediction result
        predictions.append(round(pred[0], 2))

    assert predictions is not None, "Batch prediction returned no results!"
    assert len(predictions) == len(batch_data), "Prediction count does not match input data!"
    
    # Optionally check if any prediction is invalid (e.g., None values)
    assert all(pred is not None for pred in predictions), "Some batch predictions failed!"

    print("Batch prediction test passed successfully!")

