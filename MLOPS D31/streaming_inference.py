"""Task 4: Streaming-style inference on 10 held-out records."""
import time
import joblib
import pandas as pd

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
model = joblib.load("model.pkl")
passengers = pd.read_csv("outputs/unseen_passengers.csv").head(10)

print("Streaming Inference Started...")
for i, row in passengers.iterrows():
    prediction = int(model.predict(pd.DataFrame([row[FEATURES].to_dict()]))[0])
    result = "Passenger Survived" if prediction == 1 else "Passenger Did Not Survive"
    print(f"Incoming Passenger: {i + 1}")
    print(f"Prediction: {prediction}")
    print(f"Result: {result}")
    print("-" * 40)
    if i < len(passengers) - 1:
        time.sleep(1)
print("Streaming Inference Completed!")
