import os
import sys

def test_dummy():
    assert 1 == 1

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app import app

def test_home_page():
    with app.test_client() as c:
        response = c.get('/')
        assert response.status_code == 200
        assert b"<h1>Welcome To Data Science Project Session</h1>" in response.data

def test_predict_datapoint():
    with app.test_client() as c:
        response = c.get('/predict')
        assert response.status_code == 200
        assert b"<h3> Diamond Price Prediction </h3>" in response.data

    with app.test_client() as c:
        response = c.post('/predict', data={
            "carat": 0.23,
            "depth": 61.5,
            "table": 55.0,
            "x": 3.95,
            "y": 3.98,
            "z": 2.43,
            "cut": "Ideal",
            "color": "E",
            "clarity": "SI2"
        })
        assert response.status_code == 200


