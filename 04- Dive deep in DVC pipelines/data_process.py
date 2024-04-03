import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
import dvc.api
import yaml

## For output data
OUTS_DATA_PATH = os.path.join(os.getcwd(), 'outs')
os.makedirs(OUTS_DATA_PATH, exist_ok=True)

## Read the processed DF
PROCESSED_FILE_PATH = os.path.join(OUTS_DATA_PATH, 'prepared_df.csv')
df = pd.read_csv(PROCESSED_FILE_PATH)


## To features and target
X = df.drop(columns=['Exited'], axis=1)
y = df['Exited']


def process_fn(test_size: float, seed: int):
    ## Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        shuffle=True, random_state=seed, stratify=y)

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
                            ('ohe', OneHotEncoder(drop=None, sparse_output=False))
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

    all_pipeline.fit(X_train)


    ## As I did OHE, The column number may be vary
    out_categ_cols = all_pipeline.named_transformers['categorical'].named_steps['ohe'].get_feature_names_out(categ_cols)

    X_train_final = pd.DataFrame(all_pipeline.transform(X_train), columns=num_cols + list(out_categ_cols) + ready_cols)
    X_test_final = pd.DataFrame(all_pipeline.transform(X_test), columns=num_cols + list(out_categ_cols) + ready_cols)

    ## Dumping
    X_train_final.to_csv(os.path.join(OUTS_DATA_PATH, 'processed_train_X.csv'), index=False)
    y_train.to_csv(os.path.join(OUTS_DATA_PATH, 'processed_train_y.csv'), index=False)

    X_test_final.to_csv(os.path.join(OUTS_DATA_PATH, 'processed_test_X.csv'), index=False)
    y_test.to_csv(os.path.join(OUTS_DATA_PATH, 'processed_test_y.csv'), index=False)


def main():
        
    # Using Yaml file
    with open('params.yaml') as f:
        process_params = yaml.safe_load(f)['process']

    # Or using dvc.api
    process_params = dvc.api.params_show()['process']

    seed = process_params['seed']
    test_size = process_params['test_size']
    process_fn(test_size=test_size, seed=seed)


if __name__ == '__main__':
    main()