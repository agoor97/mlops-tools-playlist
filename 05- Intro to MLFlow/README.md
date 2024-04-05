### `Components of MLFlow`
![MLflow Components](https://user-images.githubusercontent.com/26833433/274929143-05e37e72-c355-44be-a842-b358592340b7.png)
![MLflow Components](https://miro.medium.com/v2/resize:fit:1400/1*4HJRLpBjbE630Fts-UZsQg.png)

-----------------
### `Notes on MLflow Tracking`

- `mlflow.set_tracking_uri()` - Sets the tracking URI for MLflow, which can be a remote server, a database connection string, or a local directory. Defaults to 'mlruns' directory.

- `mlflow.get_tracking_uri()` - Retrieves the current tracking URI.

- `mlflow.create_experiment(name)` - Creates a new experiment and returns its ID.

- `mlflow.set_experiment()` - Sets an experiment as active, creating it if it doesn't exist.

- `mlflow.start_run()` - Starts a new run or returns the active one. Automatically called by logging functions if no active run exists.

- `mlflow.end_run(status)` - Ends the current run with an optional status.

- `mlflow.log_param()` - Logs a single key-value parameter in the active run.

- `mlflow.log_params()` - Logs a dictionary key-value parameters in the active run.

- `mlflow.log_metric()` - Logs a single key-value metric in the active run.

- `mlflow.log_metrics()` - Logs a dictionary key-value metrics in the active run.
 
- `mlflow.set_tag(key, value)` - Sets a single key-value tag in the active run.

- `mlflow.log_artifact()` - Logs a local file or directory as an artifact, optionally specifying the path within the run's artifact URI.

- `mlflow.log_artifacts()` - Logs all files in a directory as artifacts, optionally specifying the artifact path.

---------------------


