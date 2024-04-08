## `Churn Detection with use of CML & DVC Tools `
    * Using different approaches for solving imbalancing dataset.
    * Using different Algorithms also.
-------------------

### `Components of MLFlow`
![MLflow Components](https://user-images.githubusercontent.com/26833433/274929143-05e37e72-c355-44be-a842-b358592340b7.png)
![MLflow Components](https://miro.medium.com/v2/resize:fit:1400/1*4HJRLpBjbE630Fts-UZsQg.png)

------------------
## MLFLOW Registery
``` bash
# isnatll mysqlclient
pip install mysqlclient

# connect to mysql locally
mlflow server --host 0.0.0.0 --port 5050 --backend-store-uri mysql://root:123456@localhost:3306/mlflow_logs --default-artifact-root ./mlruns 
```

``` python
# in your scripts, make sure to set the tracking uri
mlflow.set_tracking_uri('http://localhost:5050')
```

``` bash
# serve the model using MLFlow Registery
mlflow models serve -m "models:/forest_best_f1/Staging" --port 8000 --env-manager=local
```

``` python
# sample data that we have used before
{
    "dataframe_split": {
        "columns": [
            "Age",
            "CreditScore",
            "Balance",
            "EstimatedSalary",
            "Gender_Male",
            "Geography_Germany",
            "Geography_Spain",
            "HasCrCard",
            "Tenure",
            "IsActiveMember",
            "NumOfProducts"
        ],
        "data": [
            [-0.7541830079917924, 0.5780143566720919, 0.11375998165198585, -0.14673040749854463, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0],
            [-0.5605884106597949, 0.753908347743766, 0.7003528882054108, 1.6923927520037099, 0.0, 1.0, 0.0, 1.0, 9.0, 1.0, 1.0],
            [0.11699268000219652, -0.3221490094005933, 0.5222180917013974, -0.8721429873346316, 1.0, 1.0, 0.0, 1.0, 5.0, 0.0, 2.0],
            [0.6977764719981892, -0.7256705183297281, -1.2170740485175422, 0.07677206232885857, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0]
        ]
    }
}
```
----------