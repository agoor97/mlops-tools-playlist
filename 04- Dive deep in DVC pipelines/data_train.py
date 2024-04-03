import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, json
from imblearn.over_sampling import SMOTE
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_curve, auc
import dvc.api
import yaml

OUTS_DATA_PATH = os.path.join(os.getcwd(), 'outs')
os.makedirs(OUTS_DATA_PATH, exist_ok=True)

## Read processed files

X_train_final = pd.read_csv(os.path.join(OUTS_DATA_PATH, 'processed_train_X.csv'))
X_test_final = pd.read_csv(os.path.join(OUTS_DATA_PATH, 'processed_test_X.csv'))

y_train = pd.read_csv(os.path.join(OUTS_DATA_PATH, 'processed_train_y.csv'))
y_test = pd.read_csv(os.path.join(OUTS_DATA_PATH, 'processed_test_y.csv'))
## Convert target DataFrames to Series
y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]



## ----------------- Imbalancing Dataset ---------------------- ##

## 1. use algorithm without taking the effect of imbalancing

## 2. prepare class_weights for solving imbalance dataset
vals_count = 1 - (np.bincount(y_train) / len(y_train))
vals_count = vals_count / np.sum(vals_count)  ## normalizing

dict_weights = {}
for i in range(2):  ## 2 classes (0, 1)
    dict_weights[i] = vals_count[i]


## 3. Using SMOTE for over sampling
over = SMOTE(sampling_strategy=0.7)
X_train_resmapled, y_train_resampled = over.fit_resample(X_train_final, y_train)


## Clear metrics.json file at the beginning
with open('metrics.json', 'w') as f:
    pass



def train_model(X_train, y_train, plot_name, n_estimators, max_depth, class_weight=None):
    """ A function to train model given the required train data """
    
    global clf_name
    
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=45, class_weight=class_weight)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test_final)
    
    ## Using f1_score
    f1_test = f1_score(y_test, y_pred_test)
    
    ## Accuracy also
    acc_test = accuracy_score(y_test, y_pred_test)
    
    clf_name = clf.__class__.__name__

    ## Plot the confusion matrix 
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cbar=False, fmt='.2f', cmap='Blues')
    plt.title(f'{plot_name}')
    plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])

    ## Save the plot locally
    plt.savefig(f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.close()  


    ## -------- ROC Curve ------------------ ##
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
    roc_auc = auc(fpr, tpr)

 
    # Plot ROC curve and save it to a file
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')  # Save ROC curve plot to a file
    plt.close()  # Close the plot to avoid displaying it

    ## We can dump these results if we want to plot 
    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv('roc_data.csv', index=False)

    ## Results
    new_results = {f'f1-score-{plot_name}': f1_test, f'accuracy-{plot_name}': acc_test}

    ## Dump to json file
    with open('metrics.json', 'a') as f:
        json.dump(new_results, f)  # Append new_results to the JSON file

    return True


def main():

    # Using Yaml file
    with open('params.yaml') as f:
        train_params = yaml.safe_load(f)['train']

    # Or using dvc.api
    train_params = dvc.api.params_show()['train']
    
    n_estimators = train_params['n_estimators']
    max_depth = train_params['max_depth']
    
    ## 1. without considering the imabalancing data
    train_model(X_train=X_train_final, y_train=y_train, plot_name='without-imbalance', 
                n_estimators=n_estimators, max_depth=max_depth, class_weight=None)

    ## 2. with considering the imabalancing data using class_weights
    train_model(X_train=X_train_final, y_train=y_train, plot_name='with-class-weights', 
                n_estimators=n_estimators, max_depth=max_depth, class_weight=dict_weights)

    ## 3. with considering the imabalancing data using oversampled data (SMOTE)
    train_model(X_train=X_train_resmapled, y_train=y_train_resampled, plot_name=f'with-SMOTE', 
                n_estimators=n_estimators, max_depth=max_depth, class_weight=None)

    ## Combine all conf matrix in one
    confusion_matrix_paths = [f'./without-imbalance.png', f'./with-class-weights.png', f'./with-SMOTE.png']

    ## Load and plot each confusion matrix
    plt.figure(figsize=(15, 5))  # Adjust figure size as needed
    for i, path in enumerate(confusion_matrix_paths, 1):
        im = Image.open(path)
        plt.subplot(1, len(confusion_matrix_paths), i)
        plt.imshow(im)
        plt.axis('off')  # Disable axis for cleaner visualization


    ## Save combined plot locally
    plt.suptitle(clf_name, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'conf_matrix.png', bbox_inches='tight', dpi=300)

    ## Delete old image files
    for path in confusion_matrix_paths:
        os.remove(path)

    ## ------------ combine dicts in metrics.json to be one dict ------------- ##
    ## Open the file and read its contents
    with open('metrics.json', 'r') as file:
        data = file.read()

    json_objects = data.split('}')
    json_objects = [obj + '}' for obj in json_objects if obj]

    combined_data = {}
    for obj in json_objects:
        obj_data = json.loads(obj)
        combined_data.update(obj_data)


    ## Dump the combined data back to the same file
    with open('metrics.json', 'w') as f:
        json.dump(combined_data, f)



if __name__ == '__main__':
    main()