import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import dvc.api

## For output data
OUTS_DATA_PATH = os.path.join(os.getcwd(), 'outs')
os.makedirs(OUTS_DATA_PATH, exist_ok=True)

## Read the Dataset
DATASET_PATH = os.path.join(os.getcwd(), 'dataset.csv')
df = pd.read_csv(DATASET_PATH)


def prepare_fn(age_threshold: int):
    """ A Function to prepare and filter given threshold for age feature"""

    ## Drop first 3 features
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    ## Filtering using Age Feature using threshold
    df.drop(index=df[df['Age'] > age_threshold].index.tolist(), axis=0, inplace=True)

    ## Dump the DF
    df.to_csv(os.path.join(OUTS_DATA_PATH, 'prepared_df.csv'), index=False)



def main():
    
    # Using Yaml file
    with open('params.yaml') as f:
        prepare_params = yaml.safe_load(f)['prepare']

    # Or using dvc.api
    prepare_params = dvc.api.params_show()['prepare']

    age_threshold = prepare_params['age_threshold']
    
    ## Call the function
    prepare_fn(age_threshold=age_threshold)


if __name__ == '__main__':
    main()