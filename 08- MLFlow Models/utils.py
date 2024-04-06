import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector


import warnings
warnings.filterwarnings('ignore')

## --------------------- Data Preparation ---------------------------- ##

## Read the Dataset
TRAIN_PATH = os.path.join(os.getcwd(), 'src', 'dataset.csv')
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


## As I did OHE, The column number may be vary
out_categ_cols = all_pipeline.named_transformers['categorical'].named_steps['ohe'].get_feature_names_out(categ_cols)

X_train_final = pd.DataFrame(all_pipeline.transform(X_train), columns=num_cols + list(out_categ_cols) + ready_cols)
X_test_final = pd.DataFrame(all_pipeline.transform(X_test), columns=num_cols + list(out_categ_cols) + ready_cols)

## Dump the X_test_procesesed
X_test_final.to_csv(os.path.join(os.getcwd(), 'X_test_prcoessed.csv'), index=False)
