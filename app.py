from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

app = Flask(__name__)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

def train_model(model_name):
    if model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=3)
    else:
        model = GaussianNB()

    model.fit(X_train, y_train)

    # Train results
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_cm = confusion_matrix(y_train, y_train_pred).tolist()
    train_report = classification_report(y_train, y_train_pred, output_dict=True)

    # Test results
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_cm = confusion_matrix(y_test, y_test_pred).tolist()
    test_report = classification_report(y_test, y_test_pred, output_dict=True)

    return model, train_acc, train_cm, train_report, test_acc, test_cm, test_report


@app.route("/", methods=["GET", "POST"])
def index():
    selected_model = "Naive Bayes"
    prediction = None

    model, train_acc, train_cm, train_report, test_acc, test_cm, test_report = train_model(selected_model)

    if request.method == "POST":
        selected_model = request.form.get("model")
        model, train_acc, train_cm, train_report, test_acc, test_cm, test_report = train_model(selected_model)

        try:
            features = [
                float(request.form["f1"]),
                float(request.form["f2"]),
                float(request.form["f3"]),
                float(request.form["f4"]),
            ]

            pred = model.predict([features])[0]
            prediction = f"{pred} ({target_names[pred].capitalize()})"
        except:
            prediction = "Invalid Input"

    return render_template(
        "index.html",
        train_acc=train_acc,
        train_cm=train_cm,
        train_report=train_report,
        test_acc=test_acc,
        test_cm=test_cm,
        test_report=test_report,
        prediction=prediction,
        selected_model=selected_model,
    )


if __name__ == "__main__":
    app.run(debug=True)
