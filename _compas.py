#!/usr/bin/env python3
# coding=utf-8

"""
Created on Wed Jun 20 16:02:46 2018
"""

import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from scipy.stats import entropy
from fairness_eval import FairnessEvaluator


from utils import *


def test_compas(data_prefix,test_prefix,exp_prefix,rho, beta, k):
    # experiments results urls
    models_enum_file                = './res/%s/%s/model_enum.txt'         % (data_prefix,exp_prefix)
    objectiveFunction_results       = './res/%s/%s/objectiveFunction.dump' % (data_prefix,exp_prefix)
    models_files                    = './res/%s/%s/%s_corels.mdl'         % (data_prefix, exp_prefix, data_prefix)

    # files for train performances
    unfairness_train_results              = './res/%s/%s/unfairness_train.dump'        % (data_prefix,exp_prefix)
    accuracy_train_results                = './res/%s/%s/accuracy_train.dump'          % (data_prefix,exp_prefix)

    # files for test performances
    unfairness_test_results              = './res/%s/%s/unfairness_test.dump'        % (data_prefix,exp_prefix)
    accuracy_test_results                = './res/%s/%s/accuracy_test.dump'          % (data_prefix,exp_prefix)
    
    #Algo
    genData(data_prefix, exp_prefix, test_prefix)
    
    train_file      = './res/%s/%s/%s_train.txt'     % (data_prefix, exp_prefix, data_prefix)
    test_file       = './res/%s/%s/%s_test.txt'     % (data_prefix, exp_prefix, data_prefix)
    
    test_corels(data_prefix, exp_prefix, test_prefix, rho, beta, k, opt='-c 2 -p 1', branch_depth=np.inf)
    
    #Récupération modèle et exportation dans fichier .txt
    adult_mdl       = joblib.load(models_files)  
    
    #Récupération prédiction et acc sur train et test set
    pred_train, acc_train = adult_mdl.predict(train_file)
    pred_test, acc_test = adult_mdl.predict(test_file)

    # saving objective function values 
    joblib.dump(adult_mdl.obj_, objectiveFunction_results, compress=9)

            
    # saving models enumaration file
    with open(models_enum_file,'w') as mdl:
        for k, pred_list in enumerate(adult_mdl.pred_description_):
            mdl.write('Model %i: accuracy train=%f; accuracy test=%f; obj:%f \n' % (k, acc_train[k], acc_test[k],adult_mdl.obj_[k]))
            for i in range(len(pred_list)):
                pred = pred_list[i][1:-1]
                pred=pred.replace('0_is_False','two_year_recid=yes')
                pred=pred.replace('0_is_True','two_year_recid=no')
                if (i == 0):
                    #if isnan(adult_mdl.rule_description_[k][i]):
                    #    continue
                    rule = adult_mdl.rule_description_[k][i].replace('{', '').replace('}', '').split(',')
                    mdl.write('IF ')
                    for j, r in enumerate(rule):
                        mdl.write(r)
                        if j < len(rule) - 1:
                            mdl.write(' AND ')
                    mdl.write(' THEN ')
                    mdl.write(pred)
                    mdl.write('\n')
                elif i == len(adult_mdl.pred_description_[k]) - 1:
                    mdl.write('ELSE ')
                    mdl.write(pred)
                    mdl.write('\n')
                else:
                    rule = adult_mdl.rule_description_[k][i].replace('{', '').replace('}', '').split(',')
                    mdl.write('ELSE IF ')
                    for j, r in enumerate(rule):
                        mdl.write(r)
                        if j < len(rule) - 1:
                            mdl.write(' AND ')
                    mdl.write(' THEN ')
                    mdl.write(pred)
                    mdl.write('\n')
            mdl.write('\n')
    
        mdl.close()
    
    # performance on training set 
    data_file = './algs/corels/data/%s/processed/_auditing_train.csv' % data_prefix
    dataset = pd.read_csv(data_file)
    unfairness_train = []
    for idx in range(adult_mdl.k_):
        decision = pred_train[:,idx].astype(int)
        dataset["score"] = decision
        ff = FairnessEvaluator(dataset["race:AfricanAmerican"], dataset["race:Caucasian"], dataset["score"])
        unfairness_train.append(ff.demographic_parity_discrimination())
    # savaing unfairness and accurracy values on training set
    joblib.dump(unfairness_train, unfairness_train_results, compress=9)
    joblib.dump(acc_train, accuracy_train_results, compress=9)


    # performance on test set 
    data_file = './algs/corels/data/%s/processed/_auditing_test.csv' % data_prefix
    dataset = pd.read_csv(data_file)
    unfairness_test = []
    for idx in range(adult_mdl.k_):
        decision = pred_test[:,idx].astype(int)
        dataset["score"] = decision
        ff = FairnessEvaluator(dataset["race:AfricanAmerican"], dataset["race:Caucasian"], dataset["score"])
        unfairness_test.append(ff.demographic_parity_discrimination())
    # savaing unfairness and accurracy values on training set
    joblib.dump(unfairness_test, unfairness_test_results, compress=9)
    joblib.dump(acc_test, accuracy_test_results, compress=9)


def test_compas_local(data_prefix,test_prefix,exp_prefix,rho, beta, k):
    # experiments results urls
    models_enum_file                = './res_local/%s/%s/model_enum.txt'         % (data_prefix,exp_prefix)
    objectiveFunction_results       = './res_local/%s/%s/objectiveFunction.dump' % (data_prefix,exp_prefix)
    models_files                    = './res_local/%s/%s/%s_corels.mdl'         % (data_prefix, exp_prefix, data_prefix)

    # files for train performances
    unfairness_train_results              = './res_local/%s/%s/unfairness_train.dump'        % (data_prefix,exp_prefix)
    accuracy_train_results                = './res_local/%s/%s/accuracy_train.dump'          % (data_prefix,exp_prefix)


    
    #Algo
    genData_local(data_prefix, exp_prefix, test_prefix)
    
    train_file      = './res_local/%s/%s/%s_train.txt'     % (data_prefix, exp_prefix, data_prefix)
    
    test_corels_local(data_prefix, exp_prefix, test_prefix, rho, beta, k, opt='-c 2 -p 1', branch_depth=np.inf)
    
    #Récupération modèle et exportation dans fichier .txt
    adult_mdl       = joblib.load(models_files)  
    
    #Récupération prédiction et acc sur train et test set
    pred_train, acc = adult_mdl.predict(train_file)

    acc_train = adult_mdl.predict_local(train_file)

    # saving objective function values 
    joblib.dump(adult_mdl.obj_, objectiveFunction_results, compress=9)

            
    # saving models enumaration file
    with open(models_enum_file,'w') as mdl:
        for k, pred_list in enumerate(adult_mdl.pred_description_):
            mdl.write('Model %i: accuracy train=%f; accuracy test=%f; obj:%f \n' % (k, acc_train[k], acc_train[k],adult_mdl.obj_[k]))
            for i in range(len(pred_list)):
                pred = pred_list[i][1:-1]
                pred=pred.replace('0_is_False','Income:>50K')
                pred=pred.replace('0_is_True','Income:<=50K')
                if (i == 0):
                    #if isnan(adult_mdl.rule_description_[k][i]):
                    #    continue
                    rule = adult_mdl.rule_description_[k][i]
                    rule = str(rule).replace('{', '').replace('}', '').split(',')
                    mdl.write('IF ')
                    for j, r in enumerate(rule):
                        mdl.write(r)
                        if j < len(rule) - 1:
                            mdl.write(' AND ')
                    mdl.write(' THEN ')
                    mdl.write(pred)
                    mdl.write('\n')
                elif i == len(adult_mdl.pred_description_[k]) - 1:
                    mdl.write('ELSE ')
                    mdl.write(pred)
                    mdl.write('\n')
                else:
                    rule = adult_mdl.rule_description_[k][i]
                    rule = str(rule).replace('{', '').replace('}', '').split(',')
                    mdl.write('ELSE IF ')
                    for j, r in enumerate(rule):
                        mdl.write(r)
                        if j < len(rule) - 1:
                            mdl.write(' AND ')
                    mdl.write(' THEN ')
                    mdl.write(pred)
                    mdl.write('\n')
            mdl.write('\n')
    
        mdl.close()
    
    # performance on training set 
    data_file = './algs/corels/data/%s/local/_auditing_train.csv' % data_prefix
    dataset = pd.read_csv(data_file)
    unfairness_train = []
    for idx, pred_list in enumerate(adult_mdl.pred_description_):
        decision = pred_train[:,idx].astype(int)
        dataset["score"] = decision
        ff = FairnessEvaluator((dataset['race:AfricanAmerican']==0).apply(int), (dataset['race:Caucasian']==1).apply(int), dataset['score'])
        unfairness_train.append(ff.demographic_parity_discrimination())
    # savaing unfairness and accurracy values on training set
    joblib.dump(unfairness_train, unfairness_train_results, compress=9)
    joblib.dump(acc_train, accuracy_train_results, compress=9)


   