import pandas as pd
import mlflow
from sklearn.metrics import f1_score
from mlflow.tracking import MlflowClient
import time

tracking_uri = 'sqlite:///mlflow.db'
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri= tracking_uri)
model_name = "Custormer-churn-models"


# client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
# runs = client.search_runs(experiment_ids='1',
#                           filter_string="metrics.test_f1_score >0.595",
#                           run_view_type=ViewType.ACTIVE_ONLY,
#                           max_results=5,
#                           order_by=["metrics.test_f1_score ASC"]
#                         )

def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.replace(' ', '_').str.lower()

    categorical_col = data.dtypes[data.dtypes == 'object'].index.tolist()
    for col in categorical_col:
        data[col] = data[col].str.replace(' ', '_').str.lower()

    data = data[data['totalcharges'] != '_']
    data['totalcharges'] = data['totalcharges'].astype('float32')
    return data

def prepare_data(data):

    data['churn'] = (data.churn=='yes').astype(int)
    categorical_col = data.dtypes[data.dtypes == 'object'].index.tolist()
    numerical_col = ['tenure', 'totalcharges', 'monthlycharges']

    categorical_col.remove('customerid')
    feature_cols = categorical_col + numerical_col

    data_x = data.drop(['churn'], axis = 1)
    data_x = data_x[feature_cols].to_dict(orient = 'records')

    data_y = data.pop('churn')
    return (data_x, data_y)

def test_model(name, stage, test_x, test_y):
    prev_time = time.time()
    model = mlflow.pyfunc.load_model(f"models:/{name}/{stage}")
    y_pred = model.predict(test_x)
    new_time = time.time()
    time_diff = new_time - prev_time
    return {"F1 Score": f1_score(test_y, y_pred), "Time interval": time_diff}


def compare_models():

    df = load_data("data/Telco-Customer-Churn.csv")
    data_x, data_y = prepare_data(df)
    production_model = test_model(name=model_name, stage="Production", test_x = data_x, test_y = data_y)
    stage_model = test_model(name=model_name, stage="Staging", test_x = data_x, test_y = data_y)

    if (stage_model['F1 Score'] >= production_model['F1 Score']):
        if stage_model['Time interval'] < production_model['F1 Score']:
            
            client.transition_model_version_stage(name=model_name,
                                          version=4,
                                          stage="Production",
                                          archive_existing_versions=True
                                        )