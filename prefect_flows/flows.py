from prefect import flow, task
from scripts.preprocess import load_and_preprocess
from scripts.train import train

@task
def preprocess_task():
    return load_and_preprocess("./data/q1_q17_dr25_koi_2025.05.14_22.18.22.csv")

@task
def train_task(X_y):
    X, y, n_c = X_y
    train(X, y, n_c)

@flow(name="Exoplanet Detection Pipeline")
def exoplanet_flow():
    data = preprocess_task()
    train_task(data)

if __name__ == "__main__":
    exoplanet_flow()
