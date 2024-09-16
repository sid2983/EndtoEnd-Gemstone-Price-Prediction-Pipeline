from __future__ import annotations
import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from src.pipeline.training_pipeline import TrainingPipeline
from datetime import datetime, timedelta

training_pipeline = TrainingPipeline()

with DAG(
    "gemstone_training_pipeline",
    default_args={
        "retries": 2,
    },
    description="DAG for training gemstone model",
    #trigger after 15 minutes
    schedule=timedelta(minutes=15),
    start_date = pendulum.datetime(2024,9,16, tz="UTC"),
    catchup=False,
    tags=["machine_learning","Regression","Gemstone"],
) as dag:
    dag.doc_md = __doc__



    data_ingestion_task = BashOperator(
        task_id="data_ingestion",
        bash_command="cd /app && dvc repro -s data_ingestion --force >> /app/logs/data_ingestion.log 2>&1",
    )


    data_ingestion_task.doc_md = dedent(
        """
        #### Ingestion Task
        This task is responsible for ingesting the data.
        
        """
    )


    data_transformation_task = BashOperator(
        task_id="data_transformation",
        bash_command="cd /app && dvc repro -s data_transformation --force >> /app/logs/data_transformation.log 2>&1",
    )

    data_transformation_task.doc_md = dedent(
        """
        #### Transformation Task
        This task is responsible for transforming the data.
        
        """
    )


    model_trainer_task = BashOperator(
        task_id="model_trainer",
        bash_command="cd /app && dvc repro -s model_training --force >> /app/logs/model_training.log 2>&1",
    )

    model_trainer_task.doc_md = dedent(
        """
        #### Model Training Task
        This task is responsible for training the model.
        
        """
    )

    push_to_s3_task = BashOperator(
        task_id="push_to_s3",
        bash_command="cd /app && dvc push >> /app/logs/dvc_push.log 2>&1",
    )

#execution flow
data_ingestion_task >> data_transformation_task >> model_trainer_task >> push_to_s3_task