## Earlier this was not giving uri in http/https form, so to check we used below code.
import mlflow
print("Printing tracking URI scheme below")
print(mlflow.get_tracking_uri())
print("\n")

## Now, for making that uri in http/https form, we are setting it explicitly in the main code as below.
import mlflow
print("Printing new tracking URI scheme below")
print(mlflow.get_tracking_uri())
print("\n")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

## We do not need to do above this bcz, the error has been resolved by mlflow team.