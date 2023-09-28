import pandas as pd
import numpy as np
import os

def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.replace(' ', '_').str.lower()
    return data

def compile(data_path):
    dataframe = pd.DataFrame(columns=['contect', 'title', 'body'])
    for mail in os.listdir(data_path):
        df_path = os.path.join(data_path, mail)
        df = load_data(df_path)
        dataframe = pd.concat([dataframe, df])
    return dataframe

def main(data_path = './data/text/'):

    dataframe = compile(data_path)
    dataframe.to_csv('./data/complete431.csv', index=False)
    print('Process Successful')

#%cd /home/godwin/extend/Documents/Workflow/Churn-Prediction-in-a-Telecom-Company
main()