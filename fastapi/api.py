import os

from fastapi import FastAPI
from minio import Minio
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature



####### DATABASE #########################################################################

DATABASE_URL = "postgresql://postgres:postgres@postgres:5432/mlflow"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def list_experiments_in_db():
    try:
        with engine.connect() as connection:
            query = text("SELECT * FROM experiments")
            result = connection.execute(query)
            experiments = []
            for row in result:
                experiments.append(dict(row._mapping))
            return experiments
    except Exception as e:
        print(e)
        return [str(e)]
    
##########################################################################################



####### MINIO ############################################################################

minio_client = Minio(
    endpoint="minio:9000",
    access_key="cw3C8ZVcTL1Z3ooJUZu7",
    secret_key="MuG9Wod7cEoyVTcDkc6dVu0CKvnCOcb75L9q6sSK",
    secure=False # True if you are using https, False if http
)


def list_buckets():
    return minio_client.list_buckets()

##########################################################################################



####### MLFLOW ##########################################################################

mlflow.set_tracking_uri("http://mlflow:5000")

def test_run():
    with mlflow.start_run():
        X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 0])
        lr = LogisticRegression()
        lr.fit(X, y)
        score = lr.score(X, y)
        mlflow.log_metric("score", score)
        predictions = lr.predict(X)
        signature = infer_signature(X, predictions)
        mlflow.sklearn.log_model(lr, "model", signature=signature)
        return {
            "runID": f"{mlflow.active_run().info.run_uuid}", 
            "run name": f"{mlflow.active_run().info.run_name}"
        }
    
def list_runs_in_experiment(experiment_name: str):
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        runs = mlflow.search_runs(experiment.experiment_id)
        return runs.to_dict(orient="records")
    except Exception as e:
        return [str(e)]


##########################################################################################



####### API ##############################################################################

app = FastAPI()

@app.get("/buckets")
def get_list_of_buckets():
    return list_buckets()

@app.get("/experiments")
def get_list_of_experiments():
    return list_experiments_in_db()

@app.get("/test_run")
def run_test():
    return test_run()

@app.get("/runs")
def get_runs_in_experiment(experiment_name: str):
    return list_runs_in_experiment(experiment_name)

##########################################################################################



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)