"""Task 2: Online inference on one held-out passenger record."""
import joblib
import pandas as pd

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

model = joblib.load("model.pkl")
data = pd.read_csv("outputs/unseen_passengers.csv")
passenger = data.loc[[0], FEATURES]
prediction = int(model.predict(passenger)[0])
print("Passenger Details:")
print(passenger.to_string(index=False))
print("\\nPredicted Value:", prediction)
print("Prediction:", "Passenger Survived" if prediction == 1 else "Passenger Did Not Survive")
