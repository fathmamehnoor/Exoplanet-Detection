import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_and_preprocess(path):
    input_length = 11   

    # If the file has comment lines at the top, use comment='#'
    df = pd.read_csv(path, comment='#')

    features = [
        "koi_disposition", "koi_period", "koi_impact", "koi_duration", "koi_depth",
        "koi_prad", "koi_teq", "koi_insol", "koi_model_snr",
        "koi_steff", "koi_slogg", "koi_srad"
    ]

    df = df[features]

    # Drop rows with any missing values
    df_clean = df.dropna()

    labels = df_clean["koi_disposition"]
    features = df_clean.drop("koi_disposition", axis=1)

    # Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    # Reshape for Conv1D
    X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))  # (4599, 11, 1)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels) 

    # Convert labels to one-hot if using softmax output
    num_classes = len(np.unique(y_encoded))
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

    return X_cnn, y_categorical, num_classes