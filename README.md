# 🏆 Diamond Price Prediction MLOps Project

![Airflow-MLFlow-Flask](https://img.shields.io/badge/MLops-Automated-blue)
![Docker](https://img.shields.io/badge/Docker-Running-green)
![AWS EC2](https://img.shields.io/badge/AWS%20EC2-Deployed-yellow)

Welcome to the **Diamond Price Prediction MLOps Project**! This project demonstrates the end-to-end automation and deployment of a machine learning pipeline using modern tools like Airflow, MLflow, Docker, DVC, and AWS EC2. It focuses on predicting diamond prices based on key features like carat, clarity, and cut.

<p align="center">
  <img src="" alt="Airflow UI Screenshot" width="700"/>
  <img src="https://drive.google.com/file/d/19VgZ7e7p0QkC_dkohFLATNVjSYZUrnEO/view?usp=sharing" alt="MLflow UI Screenshot" width="700"/>
</p>

---

## 🚀 Project Summary

This project involves building a machine learning model for predicting diamond prices and deploying it in a fully automated MLOps pipeline. The project is hosted on **AWS EC2** and leverages **Docker** to containerize multiple services such as:

- **Airflow**: To orchestrate and manage the entire machine learning workflow.
- **MLflow**: For experiment tracking and model versioning.
- **Flask**: For serving the machine learning model via a REST API.
- **DVC**: For data version control and reproducibility.

The components are designed to run seamlessly within Docker containers, making the pipeline portable and scalable.

---

## 📚 Features

### 🔄 End-to-End Pipeline Automation
- Automated data ingestion, transformation, model training, and model serving.
- Airflow DAGs handle the complete workflow for orchestrating tasks such as **data ingestion, transformation, model training, and deployment**.

### 🧬 Model Versioning with MLflow
- MLflow tracks the performance metrics, hyperparameters, and artifacts of each experiment.
- Easily switch between models and compare different runs to find the best-performing version.

### 📦 Containerized Deployment with Docker
- Entire system packaged into **Docker containers**, ensuring consistency across environments.
- Separate Docker containers for **Airflow, MLflow**, and **Flask** services.

### ☁️ AWS EC2 Hosting
- Deployed the entire pipeline on **AWS EC2** for cloud scalability.
- Data is stored and versioned in **AWS S3** using DVC, ensuring efficient data management.

### 📡 REST API for Predictions
- A **Flask** app serves the model and exposes a REST API to make predictions.
- Use `POST` requests with JSON payloads to get diamond price predictions.

---

## 🧰 Tech Stack

<p align="center p-3">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Apache%20Airflow-017CEE?logo=apacheairflow&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/DVC-945DD6?logo=dvc&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-FF9900?logo=amazonaws&logoColor=white" />
  <img src="https://img.shields.io/badge/EC2-FF9900?logo=amazonec2&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS%20S3-569A31?logo=amazons3&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?logo=githubactions&logoColor=white" />
</p>

- **Languages & Frameworks**: Python, Flask
- **Containerization**: Docker
- **Pipeline Orchestration**: Apache Airflow
- **Experiment Tracking**: MLflow
- **Data Version Control**: DVC
- **Cloud Infrastructure**: AWS EC2, S3
- **CI/CD**: GitHub Actions

---

## ⚙️ Project Structure

```bash
.
├── airflow/
├── flask_app/
├── mlruns/             # MLflow experiment tracking
├── dvc.yaml            # DVC pipeline definition
├── Dockerfile.airflow   # Docker configuration for Airflow
├── Dockerfile.flask     # Docker configuration for Flask app
├── requirements.txt     # Python dependencies
├── start.sh             # Start script for Docker services
└── tox.ini              # For running tests
