import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.externals import joblib
import numpy as np

def fidelity_unfairness(dataset, lambdas, betas, csv_file):
    frames = []
    for k in lambdas:
        for b in betas:
            experiment = '/lambda='+str(k) + '__' + 'beta='+str(b)
            
            # files for train performances
            unfairness_train_results              = './res/%s/%s/unfairness_train.dump'        %(dataset, experiment)
            accuracy_train_results                = './res/%s/%s/accuracy_train.dump'           %(dataset, experiment)

            # files for test performances
            unfairness_test_results              = './res/%s/%s/unfairness_test.dump'         %(dataset, experiment)
            accuracy_test_results                = './res/%s/%s/accuracy_test.dump'           %(dataset, experiment)

            # unfairness and fidelity on training set
            unfairness_train = joblib.load(unfairness_train_results)
            fidelity_train = joblib.load(accuracy_train_results)

            # unfairness and fidelity on test set
            unfairness_test = joblib.load(unfairness_test_results)
            fidelity_test = joblib.load(accuracy_test_results)

            df = pd.DataFrame()
            #df['fidelity_train'] = fidelity_train[:10]
            df['fidelity_train'] = fidelity_train
            df['unfairness_train'] = unfairness_train
            df['fidelity_test'] = fidelity_test
            df['unfairness_test'] = unfairness_test

            df['lambda'] = 'lambda = ' + str(k)
            df['beta'] = 'beta = ' + str(b)

            frames.append(df)

    
    
    analysis_df = pd.concat(frames)

    #analysis_df.to_csv(csv_file, encoding='utf-8', index=False)

    return analysis_df


def fidelity_unfairness_clear(dataset, lambdas, betas, csv_file):
    frames = []
    for k in lambdas:
        for b in betas:
            experiment = '/lambda='+str(k) + '__' + 'beta='+str(b)
            
            # files for train performances
            unfairness_train_results              = './res/%s/%s/unfairness_train.dump'        %(dataset, experiment)
            accuracy_train_results                = './res/%s/%s/accuracy_train.dump'           %(dataset, experiment)

            # files for test performances
            unfairness_test_results              = './res/%s/%s/unfairness_test.dump'         %(dataset, experiment)
            accuracy_test_results                = './res/%s/%s/accuracy_test.dump'           %(dataset, experiment)

            # unfairness and fidelity on training set
            unfairness_train = joblib.load(unfairness_train_results)
            fidelity_train = joblib.load(accuracy_train_results)

            # unfairness and fidelity on test set
            unfairness_test = joblib.load(unfairness_test_results)
            fidelity_test = joblib.load(accuracy_test_results)

            df = pd.DataFrame()
            #df['fidelity_train'] = fidelity_train[:10]
            df['fidelity_train'] = fidelity_train
            df['unfairness_train'] = unfairness_train
            df['fidelity_test'] = fidelity_test
            df['unfairness_test'] = unfairness_test

            df['lambda'] = 'lambda = ' + str(k)
            df['beta'] = str(b)

            frames.append(df)

    
    
    analysis_df = pd.concat(frames)

    #analysis_df.to_csv(csv_file, encoding='utf-8', index=False)

    return analysis_df



lambdas = [0.005, 0.01]
betas = [0.0, 0.1, 0.2, 0.5, 0.7, 0.9]


dataset_adult = "adult/original"

dataset_compas = "compas/original"



dataset_adult = fidelity_unfairness_clear(dataset_adult, lambdas, betas, '')
dataset_adult['which'] = 'Adult'


dataset_compas = fidelity_unfairness_clear(dataset_compas, lambdas, betas, '')
dataset_compas['which'] = 'Compas'


csv_file = "plotting_scripts/data/both.csv"
analysis_df = pd.concat([dataset_adult, dataset_compas])
analysis_df.to_csv(csv_file, encoding='utf-8', index=False)