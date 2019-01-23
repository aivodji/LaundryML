import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from fairml import audit_model
from fairml import plot_dependencies
import matplotlib.pyplot as plt


def get_audit(data_prefix, exp_prefix):

    data_file = './algs/corels/data/%s/processed/_auditing_test.csv'        % (data_prefix)
    scores_file = './algs/corels/data/%s/processed/_scores_test.csv'        % (data_prefix)
    dataset = pd.read_csv(data_file)
    scores = pd.read_csv(scores_file)
    models_files  = './res/%s/%s/%s_corels.mdl'         % (data_prefix, exp_prefix, data_prefix)
    test_file      = './res/%s/%s/%s_test.txt'     % (data_prefix, exp_prefix, data_prefix)
    unfairness_results = './res/%s/%s/unfairness_test.dump' %(data_prefix,exp_prefix)    
    accuracy_results = './res/%s/%s/accuracy_train.dump' %(data_prefix,exp_prefix)
    audit_file      = './res/%s/%s/%s_audit.png'     % (data_prefix, exp_prefix, data_prefix)
    #Récupération modèles
    adult_mdl   = joblib.load(models_files)  
    acc = joblib.load(accuracy_results)
    diff_metric = joblib.load(unfairness_results)
    distance  = []
    for i in range(len(diff_metric)):
        d = (diff_metric[i])**2 + (1-acc[i])**2
        distance.append(d)
    idx = np.argmin(distance)

    y_pred, acc = adult_mdl.predict(test_file)

    clf = LogisticRegression(penalty='l2', C=0.01)
    clf.fit(dataset.values, y_pred[:,idx].astype(int))

    #  call audit model with model
    total, _ = audit_model(clf.predict, dataset)


    # generate feature dependence plot
    fig = plot_dependencies(
        total.median(),
        reverse_values=False,
        title="FairML feature dependence",
        fig_size=(6, 9)
    )
    plt.savefig(audit_file, transparent=False, bbox_inches='tight') 


def get_audit_local(data_prefix, exp_prefix):

    data_file = './algs/corels/data/%s/local/_auditing_train.csv'        % (data_prefix)
    scores_file = './algs/corels/data/%s/local/_scores_train.csv'        % (data_prefix)
    dataset = pd.read_csv(data_file)
    scores = pd.read_csv(scores_file)
    models_files  = './res_local/%s/%s/%s_corels.mdl'         % (data_prefix, exp_prefix, data_prefix)
    train_file      = './res_local/%s/%s/%s_train.txt'     % (data_prefix, exp_prefix, data_prefix)
    
    unfairness_results = './res_local/%s/%s/unfairness_train.dump' %(data_prefix,exp_prefix)    
    accuracy_results = './res_local/%s/%s/accuracy_train.dump' %(data_prefix,exp_prefix)
    audit_file      = './res_local/%s/%s/%s_audit.png'     % (data_prefix, exp_prefix, data_prefix)
    #Récupération modèles
    adult_mdl   = joblib.load(models_files)  
    acc = joblib.load(accuracy_results)
    diff_metric = joblib.load(unfairness_results)
    distance  = []
    for i in range(len(diff_metric)):
        d = (diff_metric[i])**2 + (1-acc[i])**2
        distance.append(d)
    idx = np.argmin(distance)

    y_pred, acc = adult_mdl.predict(train_file)

    clf = LogisticRegression(penalty='l2', C=0.01)
    clf.fit(dataset.values, y_pred[:,idx].astype(int))

    #  call audit model with model
    total, _ = audit_model(clf.predict, dataset)


    # generate feature dependence plot
    fig = plot_dependencies(
        total.median(),
        reverse_values=False,
        title="FairML feature dependence",
        fig_size=(6, 9)
    )
    plt.savefig(audit_file, transparent=False, bbox_inches='tight') 



