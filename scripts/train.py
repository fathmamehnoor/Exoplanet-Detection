import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import mlflow
import mlflow.tensorflow
import os

def build_model(input_length, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_length, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train(X, y, num_classes, model_name="kepler_cnn_model", experiment_name="Kepler-Exoplanet-Classification"):
    mlflow.set_experiment(experiment_name)
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name=model_name):
        # Split data into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        input_length = X_train.shape[1]  # number of time steps / features

        # Build model
        model = build_model(input_length, num_classes)

        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Save model and log with MLflow
        os.makedirs("outputs", exist_ok=True)
        model_path = f"outputs/{model_name}.keras"
        model.save(model_path)
        mlflow.log_artifact(model_path)

        return model, history
