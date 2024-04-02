## `Churn Detection with use of CML & DVC Tools `
    * Using different approaches for solving imbalancing dataset.
    * Using different Algorithms also.
-------------------
### Note
> `cml-churn.yaml` file is attached to this directory. You can put it in `.github/workflows/cml-churn.yaml` as usual.

-----------------------------------------------------------------
## `DVC Installation`
``` bash
pip install dvc
pip install dvc-gdrive  # for using gdrive
```

### Git
``` bash
git init
git add .
git commit -m "initial commit"
git remote add origin <ssh or http link>
git push
git status
```

### DVC
``` bash
dvc init
dvc add data.csv
dvc remote add --default myremote gdrive://<fodler-id> # if using gdrive
dvc push
dvc status
dvc doctor
```


``` bash
# find the default.json file credentials
C:\Users\YourUsername\AppData\Local
```
