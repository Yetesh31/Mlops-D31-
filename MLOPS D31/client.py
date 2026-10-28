import requests

url = "http://127.0.0.1:8000/predict"
passenger = {
    "Pclass": 1,
    "Sex": "female",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 80.0,
    "Embarked": "S"
}
response = requests.post(url, json=passenger, timeout=10)
print("Status Code:", response.status_code)
print("Response:")
print(response.json())
