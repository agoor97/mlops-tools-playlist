## Versioning using Git & DVC

``` bash
# Initialize 
git init
dvc init

# Track data using DVC
dvc add dataset.csv
# Add remote for DVC (e.g: gdrive)
dvc remote add storage gdrive://<folder_id>
dvc push

git add . & git commit -m "Initial commit"
# If you want to push to GitHub
git remote add origin <ssh-or-http-link> 
git push

## Now Change something, Delete or update the dataset and also code
dvc add dataset.csv
git add . & git commit -m "chnage dataset"
dvc push

## Return to the previous version
git checkout <commit-hash>
dvc checkout

## If you want to return to the original Version of Dataset
git checkout HEAD~1 dataset.csv.dvc
dvc checkout
```
-------------------------------

## `DVC Pipelines`
* provide an efficient way to reproduce them

``` bash
dvc repro

## plots and metrics
dvc plots show roc-data.csv -x fpr -y tpr
dvc metrics show

# Get changes with main branch (for exmaple)
dvc metrics diff <commit-hash or branch-name>
dvc params diff <commit-hash or branch-name>
dvc dag # Directed Acyclic Graph
dvc metrics show

## plots
dvc plots show roc-data.csv -x fpr -y tpr

## Expeirements -- without commiting
dvc exp run
dvc exp diff <exp-1> <exp-2>
```
--------------------
### Note
> If you want: `cml-churn.yaml` file is attached to this directory. You can put it in `.github/workflows/cml-churn.yaml` as usual.
