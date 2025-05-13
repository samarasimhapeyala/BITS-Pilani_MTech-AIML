import pickle
import numpy as np
import time
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

# Load the trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Add Prometheus metrics
instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app)

@app.get("/")
def read_root():
    return {"message": "Fashion-MNIST Model is running!"}

@app.get("/predict/")
def predict():
    start_time = time.time()  # Start timing

    # Generate synthetic test image in the correct shape: (1, 28, 28, 1)
    synthetic_test_image = np.random.rand(1, 28, 28, 1).astype(np.float32)

    try:
        # Make prediction
        prediction = model.predict(synthetic_test_image)
        predicted_label = int(np.argmax(prediction))
        
        latency = time.time() - start_time  # Calculate latency

        # Print metrics for visibility
        print("\n--- Request Metrics ---")
        print(f"Prediction: {predicted_label}")
        print(f"Latency (seconds): {latency:.6f}\n")

        return {"prediction": predicted_label, "latency_seconds": latency}

    except Exception as e:
        print("\nError during prediction:", e)
        return {"error": str(e)}
