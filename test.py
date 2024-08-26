import sys
import mlflow
import os
# from src.logger.loggings import logging

# logging.info("This is my first ml package testing")

# print(loggings.t)

# print(sys.exc_info())

# def calc(x,y,ops=None):
#     if ops=='add':
#         return x+y
#     elif ops=='sub':
#         return x-y
#     elif ops=='mul':
#         return x*y
#     elif ops=='div':
#         return x/y
#     else:
#         return "Invalid Operation"
    

import numpy as np
import pandas as pd

Data = {
    'Name': ['Tom', 'nick', 'krish', 'jack'],
    'Age': [20, 21, 19, 18],
    'City': ['Hyderabad', 'Bangalore', 'Chennai', 'Mumbai'],
    'Salary': [10000, 20000, 15000, 30000],
    'Designation': ['Software Engineer', 'Data Scientist', 'Data Analyst', 'ML Engineer']
}

df = pd.DataFrame(Data)


#create a function for this

def save_to_csv(df, path):
    df.to_csv(path, index=False)
    return "Data Saved Successfully"



if __name__=="__main__":
    os.makedirs("tdata/raw", exist_ok=True)
    path = "tdata/raw/data.csv"
    
    result = save_to_csv(df, path)
    print(result)




# if __name__=="__main__":
    # x=934
    # y=1868
    # with mlflow.start_run():
    #     result = calc(x,y,ops='add')
    #     mlflow.log_param("x",x) 
    #     mlflow.log_param("y",y)
    #     mlflow.log_param("operation","add")
    #     mlflow.log_metric("result",result)

    #     print("Addition Result:",result)



