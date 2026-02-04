## This is a sample code to demonstrate mlflow tracking with DagsHub integration - For setting up the remote tracking server in DagsHub

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

## DagsHub integration
import dagshub
dagshub.init(repo_owner='AmitGour-03', repo_name='MLOPS-Experiments-with-MLFlow', mlflow=True)

# Set the tracking uri to DagsHub MLflow server (from Dagshub)
mlflow.set_tracking_uri("https://dagshub.com/AmitGour-03/MLOPS-Experiments-with-MLFlow.mlflow")

## Above three lines of code is for setting up the remote tracking server in DagsHub

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 10

# Mention your experiment below o/w each and every run will be logged under 'Default' experiment
mlflow.set_experiment('YT-MLOPS-Exp2')

with mlflow.start_run():  ## we can mention experiment id here by creating in mlflow ui which helps us to avoid set_experiment line (above)
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    # log artifacts (file & image) using mlflow
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)  ## here file name does not matter, it will pick the current file.

    # tags
    mlflow.set_tags({"Author": 'Amit', "Project": "Wine Classification"})

    # Log the model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")

    print(accuracy)