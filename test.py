import sys
import mlflow
from src.logger.loggings import logging

logging.info("This is my first ml package testing")

# print(loggings.t)

# print(sys.exc_info())

def calc(x,y,ops=None):
    if ops=='add':
        return x+y
    elif ops=='sub':
        return x-y
    elif ops=='mul':
        return x*y
    elif ops=='div':
        return x/y
    else:
        return "Invalid Operation"
    


if __name__=="__main__":
    x=934
    y=1868
    with mlflow.start_run():
        result = calc(x,y,ops='add')
        mlflow.log_param("x",x) 
        mlflow.log_param("y",y)
        mlflow.log_param("operation","add")
        mlflow.log_metric("result",result)

        print("Addition Result:",result)

