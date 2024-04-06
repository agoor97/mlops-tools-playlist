## main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, argparse
from imblearn.over_sampling import SMOTE
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_curve, auc
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

## --------------------- Data Preparation ---------------------------- ##

## Read the Dataset
TRAIN_PATH = os.path.join(os.getcwd(), 'dataset.csv')
df = pd.read_csv(TRAIN_PATH)

## Drop first 3 features
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

## Filtering using Age Feature using threshold
df.drop(index=df[df['Age'] > 80].index.tolist(), axis=0, inplace=True)


## To features and target
X = df.drop(columns=['Exited'], axis=1)
y = df['Exited']

## Split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=45, stratify=y)


## --------------------- Data Processing ---------------------------- ##

## Slice the lists
num_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
categ_cols = ['Gender', 'Geography']

ready_cols = list(set(X_train.columns.tolist()) - set(num_cols) - set(categ_cols))


## For Numerical
num_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(num_cols)),
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ])


## For Categorical
categ_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(categ_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('ohe', OneHotEncoder(drop='first', sparse_output=False))
                    ])


## For ready cols
ready_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(ready_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent'))
                    ])



## combine all
all_pipeline = FeatureUnion(transformer_list=[
                                    ('numerical', num_pipeline),
                                    ('categorical', categ_pipeline),
                                    ('ready', ready_pipeline)
                                ])

## apply
X_train_final = all_pipeline.fit_transform(X_train)
X_test_final = all_pipeline.transform(X_test)


## --------------------- Impalancing ---------------------------- ##

# 1. use algorithm without taking the effect of imbalancing

## 2. prepare scale_pos_weight in xgboost
scale = np.bincount(y_train)[0]/np.bincount(y_train)[1]

## 3. Using SMOTE for over sampling
over = SMOTE(sampling_strategy=0.7)
X_train_resmapled, y_train_resampled = over.fit_resample(X_train_final, y_train)


## --------------------- Modeling ---------------------------- ##

def train_model(X_train, y_train, plot_name, n_estimators: int, 
                learning_rate: float, max_depth: int, scale=None):

    mlflow.set_experiment(f'churn-detection')
    with mlflow.start_run() as run:
        mlflow.set_tag('clf', 'xgboost')

        # https://xgboost.readthedocs.io/en/latest/parameter.html#:~:text=scale_pos_weight%20%5Bdefault%3D1,%2C%20py3.
        # Try xgboost
        clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=n_estimators, 
                                learning_rate=learning_rate,
                                max_depth=max_depth, scale_pos_weight=scale)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test_final)
        
        ## metrics
        f1_test = f1_score(y_test, y_pred_test)
        acc_test = accuracy_score(y_test, y_pred_test)

        # Log params, metrics, and model 
        mlflow.log_params({'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth})
        mlflow.log_metrics({'accuracy': acc_test, 'f1-score': f1_test})
        mlflow.xgboost.log_model(clf, f'{clf.__class__.__name__}/{plot_name}')

        ## Plot the confusion matrix and save it to mlflow
        plt.figure(figsize=(10, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cbar=False, fmt='.2f', cmap='Blues')
        plt.title(f'{plot_name}')
        plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
        plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])

        # Save the plot to MLflow
        conf_matrix_fig = plt.gcf()
        mlflow.log_figure(figure=conf_matrix_fig, artifact_file=f'{plot_name}_conf_matrix.png')
        plt.close()


        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_test)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve and save it to mlflow
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        # Save the plot to MLflow
        roc_fig = plt.gcf()
        mlflow.log_figure(figure=roc_fig, artifact_file=f'{plot_name}_roc_curve.png')
        plt.close()





def main(n_estimators: int, learning_rate: float, max_depth: int):

    # ---------------- Calling the above function -------------------- ##

    ## 1. without considering the imabalancing data
    train_model(X_train=X_train_final, y_train=y_train, plot_name='without-imbalance', 
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, scale=None)

    ## 2. with considering the imabalancing data using class_weights
    train_model(X_train=X_train_final, y_train=y_train, plot_name='with-class-weights', 
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, scale=scale)

    ## 3. with considering the imabalancing data using oversampled data (SMOTE)
    train_model(X_train=X_train_resmapled, y_train=y_train_resampled, plot_name=f'with-SMOTE', 
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, scale=None)




if __name__ == '__main__':
    ## Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=350)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1)
    parser.add_argument('--max_depth', '-d', type=int, default=15)
    args = parser.parse_args()

    ## Call the main function
    main(n_estimators=args.n_estimators, learning_rate=args.learning_rate, max_depth=args.max_depth)