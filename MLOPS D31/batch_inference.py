"""Task 3: Batch inference on 20 held-out passenger records."""
import joblib
import pandas as pd

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
model = joblib.load("model.pkl")
passengers = pd.read_csv("outputs/unseen_passengers.csv").head(20)
predictions = model.predict(passengers[FEATURES]).astype(int)
result = passengers[FEATURES].copy()
result["Predicted_Survived"] = predictions
result.to_csv("batch_predictions.csv", index=False)
print(result.to_string(index=False))
print("\\nbatch_predictions.csv created successfully!")
