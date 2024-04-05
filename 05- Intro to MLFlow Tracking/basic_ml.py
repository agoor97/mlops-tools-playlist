import mlflow
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


DATASET_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(DATASET_URL, sep=';')

X = df.drop(columns=['quality'])
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return acc, f1


def main(n_estimators, max_depth):

    mlflow.set_experiment('basic_ml')
    with mlflow.start_run():
        # Log params
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)

        mlflow.set_tag('clf', 'forest') # if you want to set a tag

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=45)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        acc, f1 = evaluate(y_test, y_pred_test)

        # Log metrics
        mlflow.log_metrics({'Accuracy': acc, 'F1-score': f1})
        # Log model
        mlflow.sklearn.log_model(clf, clf.__class__.__name__)
        


if __name__ == '__main__':
    # Take input from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=350)
    parser.add_argument('--max_depth', '-d', type=int, default=15)
    parsed_args = parser.parse_args()

    # Call the main function
    main(n_estimators=parsed_args.n_estimators, max_depth=parsed_args.max_depth)
