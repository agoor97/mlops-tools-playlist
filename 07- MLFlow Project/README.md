## `Churn Detection with use of CML & DVC Tools `
    * Using different approaches for solving imbalancing dataset.
    * Using different Algorithms also.
-------------------

### `Components of MLFlow`
![MLflow Components](https://user-images.githubusercontent.com/26833433/274929143-05e37e72-c355-44be-a842-b358592340b7.png)
![MLflow Components](https://miro.medium.com/v2/resize:fit:1400/1*4HJRLpBjbE630Fts-UZsQg.png)

------------------
## MLFLOW PROJECT
``` bash
# Get conda channels
conda config --show channels

# Build a MLFlow project, if you use one entry point with name (main)
mlflow run . --experiment-name <exp-name> # here it is {chrun-detection}

# If you have multiple entry points
mlflow run -e forest . --experiment-name churn-detection
mlflow run -e logistic . --experiment-name churn-detection
mlflow run -e xgboost . --experiment-name churn-detection

# If you want some params instead of default values
mlflow run -e logistic . --experiment-name churn-detection -P c=3.5 -P p="l2"
mlflow run -e xgboost . --experiment-name churn-detection -P n=250 -P lr=0.15 -P d=22

# run from github
mlflow run -e forest https://github.com/agoor97/project-t --experiment-name churn-detection -P n=210 -P d=17
mlflow run -e logistic git@github.com:agoor97/project-t.git --experiment-name churn-detection -P c=1.5 -P p="l2"

```