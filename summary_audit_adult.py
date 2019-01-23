import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from fairml import audit_model
from fairml import plot_dependencies
import seaborn as sns

import matplotlib.pyplot as plt


sns.set_style('whitegrid')

def get_audit(data_prefix, exp_prefix, model_id):

    data_file = './algs/corels/data/%s/processed/_auditing_train.csv'        % (data_prefix)
    dataset = pd.read_csv(data_file)

    models_files  = './res/%s/%s/%s_corels.mdl'         % (data_prefix, exp_prefix, data_prefix)
    train_file      = './res/%s/%s/%s_train.txt'     % (data_prefix, exp_prefix, data_prefix)

    audit_file      = './res/%s/%s/%s_audit.png'     % (data_prefix, exp_prefix, data_prefix)

    #Récupération modèles
    adult_mdl   = joblib.load(models_files)  
    
    y_pred, acc = adult_mdl.predict(train_file)

    clf = LogisticRegression(penalty='l2', C=0.01)
    clf.fit(dataset.values, y_pred[:,model_id].astype(int))

    #  call audit model with model
    total, _ = audit_model(clf.predict, dataset)

    return total



data_prefix = "adult"


data_file = './algs/corels/data/%s/processed/_auditing_train.csv'        % (data_prefix)
scores_file = './algs/corels/data/%s/processed/_scores_train.csv'        % (data_prefix)

dataset = pd.read_csv(data_file)
scores = pd.read_csv(scores_file)
 
# Train simple model
clf = LogisticRegression(penalty='l2', C=0.01)
clf.fit(dataset.values, scores)

print(len(list(dataset)))

#  call audit model with model
total_original, _ = audit_model(clf.predict, dataset)


exp_prefix = "original/lambda=0.005__beta=0.5"
total_found = get_audit(data_prefix, exp_prefix, 47)

save_bbox = "plotting_scripts/graphs/audit_adult_bbox.eps"
save_badml = "plotting_scripts/graphs/audit_adult_badml.eps"



fig = plot_dependencies(
        total_original.median(),
        pos_color="#b71c1c",
        negative_color="#1b5e20",
        reverse_values=False,
        #title="FairML feature dependence",
        fig_size=(3, 6)
    )
plt.savefig(save_bbox, transparent=False, bbox_inches='tight')

fig = plot_dependencies(
        total_found.median(),
        pos_color="#b71c1c",
        negative_color="#1b5e20",
        reverse_values=False,
        #title="FairML feature dependence",
        fig_size=(3, 6)
    )
plt.savefig(save_badml, transparent=False, bbox_inches='tight')

