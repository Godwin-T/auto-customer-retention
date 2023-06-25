#Importing Libraries
print('Importing Libraries')
import pandas as pd
import numpy as np

import mlflow
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


def evaluation(y_true, y_pred):

    accuracy_ = accuracy_score(y_true, y_pred)
    precision_ = precision_score(y_true, y_pred)
    recall_ = recall_score(y_true, y_pred)
    f1score_ = f1_score(y_true, y_pred)
    
    out = {"Accuracy Score" : accuracy_, 
        "Precision Score" :precision_, 
        "Recall Score" : recall_, 
        "F1 Score" : f1score_}
    return out


#Set mlflow
print('Settomg mlflow')
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('Telcom Churn')

#Load Data
def data_loader(path):

    print('Loading data')
    data = pd.read_csv(path)
    data.columns = data.columns.str.replace(' ', '_').str.lower()

    categorical_col = data.dtypes[data.dtypes == 'object'].index.tolist()
    for col in categorical_col:
        data[col] = data[col].str.replace(' ', '_').str.lower()
    return data

#Data Preparation
def data_prep(data):

    print('Data preparation')
    data = data[data['totalcharges'] != '_']
    data['totalcharges'] = data['totalcharges'].astype('float32')

    data['churn'] = (data['churn']=='yes').astype(int)
    categorical_col = data.dtypes[data.dtypes == 'object'].index.tolist()
    numerical_col = ['tenure', 'totalcharges', 'monthlycharges']

    categorical_col.remove('customerid')

    #Spliting data
    print('Spliting data')
    train_data, test_data = train_test_split(data, test_size=0.25,
                                            random_state=0)
    train_x = train_data.drop(['churn'], axis = 1)
    test_x = test_data.drop(['churn'], axis = 1)

    train_y = train_data.pop('churn')
    test_y = test_data.pop('churn')
    output = (train_x, train_y, test_x, test_y)
    feature_cols = categorical_col + numerical_col
    return output, feature_cols

#Encoding features
def encode_data(data, columns):

    print('Encoding features')
    (train_x, train_y, test_x, test_y) = data
    dv = DictVectorizer(sparse = False)
    dv.fit(train_x[columns].to_dict(orient = 'records'))

    train_x = dv.transform(train_x[columns].to_dict(orient = 'records'))
    test_x = dv.transform(test_x[columns].to_dict(orient = 'records'))
    output = (train_x, train_y, test_x, test_y)
    return dv, output

#Model training
def model_training(data):

    print('Training model')
    (train_x, train_y, test_x, test_y) = data

    model = LogisticRegression(C = 61)
    model.fit(train_x, train_y)
    return model

#Model evaluation
def model_evaluation(model, data):

    print('Evaluating model')
    (train_x, train_y, test_x, test_y) = data

    train_pred = model.predict(train_x)
    train_output_eval = evaluation(train_y, train_pred)

    test_pred = model.predict(test_x)
    test_output_eval = evaluation(test_y, test_pred) 

    print('Training data evaluation')  
    print(train_output_eval)
    print('========================================')
    print('Testing data evaluation')
    print(test_output_eval)

def main_flow(path):

    data = data_loader(path)
    data, feature_cols = data_prep(data)
    encoder, data = encode_data(data, feature_cols)
    model = model_training(data)
    model_evaluation(model, data)
    return model, encoder

    

path = './data/Telco-Customer-Churn.csv'
model, encoder = main_flow(path)

#Saving model
print('Saving model')
with open('Churn.bin', 'wb') as f:
    pickle.dump((model, encoder), f)
print('Model saved successfully')