"""Task 1: Train, evaluate, and save a preprocessing + classifier pipeline."""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical = ["Sex", "Embarked"]

df = pd.read_csv("train.csv")
X, y = df[FEATURES], df["Survived"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
preprocessing = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
])
model = Pipeline([
    ("preprocessor", preprocessing),
    ("classifier", RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced"
    )),
])
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
joblib.dump(model, "model.pkl")
print(f"Test accuracy: {accuracy:.4f}")
print("Saved complete pipeline to model.pkl")
