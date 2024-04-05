import mlflow
import os
import argparse
import time

def eval(param_1, param_2):
    return (param_1 + param_2) / 2


table_dict = {
    'a': [10, 15, 22, 70],
    'b': [15, 20, 30, 40]
            }

def main(param_1, param_2):
    mlflow.set_experiment(experiment_name='demo-Testing')
    # with mlflow.start_run(run_name='exmaple-demo') as run:
    with mlflow.start_run() as run:
        mlflow.set_tag('version', '1.0.0')

        # Log params and metric
        mlflow.log_param('param_1', param_1)
        mlflow.log_param('param_2', param_2)
        mlflow.log_metric('mean', eval(param_1=param_1, param_2=param_2))

        # Log Artifacts
        # Dummy exmaple for creating and artifact
        os.makedirs('dummy-folder', exist_ok=True)
        with open('dummy-folder/exmaple.txt', 'w') as f:
            f.write(f'Artifcat Created at: {time.asctime()}')
        mlflow.log_artifacts(local_dir='dummy-folder')

        # Log table as json artifact
        mlflow.log_table(data=table_dict, artifact_file='data_table.json')



if __name__ == '__main__':
    ## Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_1', '-p1', type=int, default=10)
    parser.add_argument('--param_2', '-p2', type=int, default=20)
    args = parser.parse_args()

    ## Call the main function
    main(param_1=args.param_1, param_2=args.param_2)