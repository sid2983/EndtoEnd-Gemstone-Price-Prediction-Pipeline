#!bin/sh
nohup airflow scheduler &
airflow webserver &

echo "Waiting for MLflow initialization..."

while [ ! -d "/app/mlruns" ]; do
    echo "MLflow directory not found. Checking again in 10 seconds..."
    sleep 10
done

echo "MLflow initialized"

mlflow ui --host 0.0.0.0 --port 5050