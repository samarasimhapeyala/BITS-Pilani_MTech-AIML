import mlflow
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import ks_2samp

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize data to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple CNN model
def create_model(input_shape=(28,28,1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and log multiple models, saving the best one
def train_and_log_models():
    best_accuracy = 0
    best_model_path = "best_model.pkl"  # Save best model as .pkl

    configurations = [
        {"epochs": 10, "batch_size": 64},
        {"epochs": 20, "batch_size": 32},
        {"epochs": 30, "batch_size": 128},
        {"epochs": 20, "batch_size": 64},  # Additional model configuration
        {"epochs": 30, "batch_size": 32}   # Additional model configuration
    ]
    
    for idx, config in enumerate(configurations):
        model_name = f"model_run_{idx + 1}"
        epochs, batch_size = config["epochs"], config["batch_size"]

        print(f"\nTraining Model: {model_name} | Epochs: {epochs} | Batch Size: {batch_size}")

        with mlflow.start_run(run_name=model_name):
            model = create_model()
            model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test.reshape(-1, 28, 28, 1), y_test))
            
            # Evaluate the model
            y_pred = np.argmax(model.predict(x_test.reshape(-1, 28, 28, 1)), axis=-1)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)

            print(f"Test Accuracy for {model_name}: {accuracy:.4f}")
            print(f"Confusion Matrix:\n{conf_matrix}")
            
            # Log evaluation metrics
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", class_report["weighted avg"]["precision"])
            mlflow.log_metric("recall", class_report["weighted avg"]["recall"])
            mlflow.log_metric("f1_score", class_report["weighted avg"]["f1-score"])

            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                with open(best_model_path, "wb") as f:
                    pickle.dump(model, f)

    print(f"\nBest model saved as: {best_model_path} with Accuracy: {best_accuracy:.4f}\n")

# Load best model and check for drift
def detect_drift():
    best_model_path = "best_model.pkl"

    with open(best_model_path, "rb") as f:
        model = pickle.load(f)

    # Function to create artificial drifted data
    def create_drifted_data(x, shift_value=0.3):
        return np.clip(x + shift_value, 0, 1)

    # Function for Kolmogorov-Smirnov drift detection
    def detect_drift_ks(x_baseline, x_new):
        ks_stats = [ks_2samp(x_baseline[:, i].flatten(), x_new[:, i].flatten()).statistic for i in range(x_baseline.shape[1])]
        return np.mean(ks_stats)

    # First, check drift on the original test data (should not detect drift)
    ks_stat_original = detect_drift_ks(x_test, x_test)
    threshold = 0.2  # Threshold for drift detection

    print(f"KS Statistic on original test data : {ks_stat_original:.4f}\n")

    with mlflow.start_run(run_name="Drift_Detection_Original"):
        mlflow.log_metric("ks_statistic_original", ks_stat_original)

    if ks_stat_original > threshold:
        print("Drift detected on original data (unexpected). There might be an issue.\n")
    else:
        print("No drift detected on original data, as expected.\n")

    # Now, create artificial drifted data and test for drift
    x_test_drifted = create_drifted_data(x_test)
    ks_stat_drifted = detect_drift_ks(x_test, x_test_drifted)

    print(f"KS Statistic on artificially drifted data: {ks_stat_drifted:.4f}\n")

    with mlflow.start_run(run_name="Drift_Detection_Artificial"):
        mlflow.log_metric("ks_statistic_drifted", ks_stat_drifted)

    if ks_stat_drifted > threshold:
        print("Drift detected on artificially generated data. Retraining model...\n")

        # Retrain the model with artificially drifted data
        with mlflow.start_run(run_name="Retraining_After_Drift"):
            model = create_model()
            model.fit(x_test_drifted.reshape(-1, 28, 28, 1), y_test, epochs=20, batch_size=64)

            # Save the retrained model
            retrained_model_path = "retrained_model.pkl"
            with open(retrained_model_path, "wb") as f:
                pickle.dump(model, f)

            print(f"Retrained model saved as: {retrained_model_path}\n")

            mlflow.log_metric("retrained", 1)
    else:
        print("No drift detected. Model remains unchanged.\n")
        mlflow.log_metric("retrained", 0)

# Run the entire pipeline
if __name__ == "__main__":
    mlflow.set_experiment("fashion_mnist_drift_detection_task")

    print("Starting initial model training...\n")
    train_and_log_models()

    print("\nLoading best model and checking for drift with original and synthetic data...\n")
    detect_drift()

