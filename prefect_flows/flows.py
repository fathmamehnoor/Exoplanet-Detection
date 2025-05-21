from prefect import flow, task
from scripts.preprocess import load_and_preprocess
from scripts.train import train
from pathlib import Path

def get_data_file():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the 'data/' directory.")
    elif len(csv_files) > 1:
        raise ValueError("Multiple CSV files found in 'data/'. Please keep only one.")
    
    return str(csv_files[0])

@task
def preprocess_task(path: str):
    return load_and_preprocess(path)

@task
def train_task(X_y):
    X, y, n_c = X_y
    train(X, y, n_c)

@flow(name="Exoplanet Detection Pipeline")
def exoplanet_flow():
    path = get_data_file()
    data = preprocess_task(path)
    train_task(data)

if __name__ == "__main__":
    
    exoplanet_flow()
