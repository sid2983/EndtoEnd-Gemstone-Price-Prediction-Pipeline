#!bin/sh
nohup airflow scheduler &
airflow webserver &

echo "Pushing DVC artifacts to S3..."
cd /app && dvc push

echo "DVC artifacts pushed to S3"


echo "Waiting for MLflow initialization..."

while [ ! -d "/app/mlruns" ]; do
    echo "MLflow directory not found. Checking again in 10 seconds..."
    sleep 10
done

echo "MLflow initialized"

mlflow ui --host 0.0.0.0 --port 5050

echo "MLflow UI started"



