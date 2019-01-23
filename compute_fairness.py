import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from fairml import audit_model
from fairml import plot_dependencies
import seaborn as sns

import matplotlib.pyplot as plt

from fairness_eval import FairnessEvaluator


def compute_fairness(dataset, data_prefix, expe_folder, experiment):
    demParity = []
    models_files  = './res/%s/%s/%s_corels.mdl'         % (expe_folder, experiment, data_prefix)
    train_file      = './res/%s/%s/%s_train.txt'     % (expe_folder, experiment, data_prefix)
    demParity_results = './res/%s/%s/demParity.dump' %(expe_folder, experiment)

    #Récupération modèles
    adult_mdl   = joblib.load(models_files)  
    
    y_pred, acc = adult_mdl.predict(train_file)
    for idx in range(adult_mdl.k_):
            decision = y_pred[:,idx].astype(int)
            dataset["score"] = decision
            ff = FairnessEvaluator(dataset["gender:Female"], dataset["gender:Male"], dataset["score"])
            demParity.append(ff.demographic_parity_discrimination())

    joblib.dump(demParity, demParity_results, compress=9)


    
experiments = []

#betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#lambdak = [0.005, 0.0075, 0.01]

betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
lambdak = [0.005]

for _lambdak in lambdak:
    for _beta in betas:
        folder = 'lambda='+str(_lambdak) + '__' + 'beta='+str(_beta)
        experiments.append(folder) 


data_file = './algs/corels/data/adult_final/processed/_auditing.csv'  
dataset = pd.read_csv(data_file)
 
expe_folder = "adult_final/original"

data_prefix = "adult_final"


for experiment in experiments:
    compute_fairness(dataset,data_prefix, expe_folder, experiment) 
