from train import train
from preprocess import load_and_preprocess

X, y, num_classes = load_and_preprocess("./data/q1_q17_dr25_koi_2025.05.14_22.18.22.csv")


model, history = train(X, y, num_classes)
