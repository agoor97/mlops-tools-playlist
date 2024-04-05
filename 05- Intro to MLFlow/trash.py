import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

print(mlflow.get_tracking_uri())


try:
    exp_id = mlflow.create_experiment('test')
    print(exp_id)
except:
    pass

# set and exp name to create it and make it active
exp = mlflow.set_experiment('test')
print(exp)

# Choose a name for run if you want
with mlflow.start_run(run_name='testsomething') as run:
    pass

# make mlflow create its fancy name
with mlflow.start_run() as run:
    mlflow.set_tag('version', '1.0.0')
    pass


# Log params
mlflow.log_param('param_1', 10)
mlflow.log_param('param_2', 20)

# Log metric
mlflow.log_metric('accuracy', 0.8)

# To get experiment ID
exp_name = mlflow.get_experiment_by_name(name='ahmed')
# Check if experiment exists
if exp_name:
    print(exp_name.experiment_id)  

# To get run id

