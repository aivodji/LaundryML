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
from fairness_eval import ConfusionMatrix, FairnessMetric
from collections import namedtuple


from utils import *

class LaundryML(namedtuple('LaundryML', 'data_prefix test_prefix exp_prefix res_folder k opt rho beta metric maj_pos min_pos sensitve_attr non_sensitve_attr decision_attr')):
        def run(self):
            # experiments results urls
            models_enum_file                = './%s/%s/%s/model_enum.txt'         % (self.res_folder, self.data_prefix, self.exp_prefix)
            objectiveFunction_results       = './%s/%s/%s/objectiveFunction.dump' % (self.res_folder, self.data_prefix, self.exp_prefix)
            models_files                    = './%s/%s/%s/%s_corels.mdl'         % (self.res_folder, self.data_prefix, self.exp_prefix, self.data_prefix)

            # files for train performances
            unfairness_train_results              = './%s/%s/%s/unfairness_train.dump'        % (self.res_folder, self.data_prefix, self.exp_prefix)
            accuracy_train_results                = './%s/%s/%s/accuracy_train.dump'          % (self.res_folder, self.data_prefix, self.exp_prefix)

            # files for test performances
            unfairness_test_results              = './%s/%s/%s/unfairness_test.dump'        % (self.res_folder, self.data_prefix, self.exp_prefix)
            accuracy_test_results                = './%s/%s/%s/accuracy_test.dump'          % (self.res_folder, self.data_prefix, self.exp_prefix)

            # generate input files
            genData(self.data_prefix, self.exp_prefix, self.test_prefix, self.res_folder)
            train_file      = './%s/%s/%s/%s_train.txt'     % (self.res_folder, self.data_prefix, self.exp_prefix, self.data_prefix)
            test_file       = './%s/%s/%s/%s_test.txt'     % (self.res_folder, self.data_prefix, self.exp_prefix, self.data_prefix)


            # call of corels' enumerator
            test_corels(self.data_prefix, self.exp_prefix, self.test_prefix, self.res_folder, rho=self.rho, beta=self.beta, k=self.k, opt=self.opt, metric=self.metric, min_pos=self.min_pos, maj_pos=self.maj_pos, branch_depth=np.inf)

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
                        #pred=pred.replace('0_is_False', self.prediction + '=Yes')
                        #pred=pred.replace('0_is_True', self.prediction + '=No')
                        #pred=pred.replace('1_is_False', self.prediction + '=No')
                        #pred=pred.replace('1_is_True', self.prediction + '=Yes')
                        if (i == 0):
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
            data_file = './data/%s/processed/_auditing_train.csv' % self.data_prefix
            dataset = pd.read_csv(data_file)
            thruth_file = './data/%s/processed/_scores_train.csv' % self.data_prefix
            thruth_dataset = pd.read_csv(thruth_file)

            unfairness_train = []

            for idx, pred_list in enumerate(adult_mdl.pred_description_):
                decision = pred_train[:,idx].astype(int)
                dataset[self.decision_attr] = decision
                cm = ConfusionMatrix(dataset[self.sensitve_attr], dataset[self.non_sensitve_attr], dataset[self.decision_attr], thruth_dataset[self.decision_attr])
                cm_majority, cm_minority  = cm.get_matrix()
                fm = FairnessMetric(cm_majority, cm_minority)
                if (self.metric == 1):
                    unfairness_train.append(fm.statistical_parity())
                if (self.metric == 2):
                    unfairness_train.append(fm.predictive_parity())
                if (self.metric == 3):
                    unfairness_train.append(fm.predictive_equality())
                if (self.metric == 4):
                    unfairness_train.append(fm.equal_opportunity())

            # savaing unfairness and accurracy values on training set
            joblib.dump(unfairness_train, unfairness_train_results, compress=9)
            joblib.dump(acc_train, accuracy_train_results, compress=9)


            # performance on test set 
            data_file = './data/%s/processed/_auditing_test.csv' % self.data_prefix
            dataset = pd.read_csv(data_file)
            thruth_file = './data/%s/processed/_scores_test.csv' % self.data_prefix
            thruth_dataset = pd.read_csv(thruth_file)

            unfairness_test = []
            for idx, pred_list in enumerate(adult_mdl.pred_description_):
                decision = pred_test[:,idx].astype(int)
                dataset[self.decision_attr] = decision
                cm = ConfusionMatrix(dataset[self.sensitve_attr], dataset[self.non_sensitve_attr], dataset[self.decision_attr], thruth_dataset[self.decision_attr])
                cm_majority, cm_minority  = cm.get_matrix()
                fm = FairnessMetric(cm_majority, cm_minority)
                if (self.metric == 1):
                    unfairness_test.append(fm.statistical_parity())
                if (self.metric == 2):
                    unfairness_test.append(fm.predictive_parity())
                if (self.metric == 3):
                    unfairness_test.append(fm.predictive_equality())
                if (self.metric == 4):
                    unfairness_test.append(fm.equal_opportunity())

            # savaing unfairness and accurracy values on training set
            joblib.dump(unfairness_test, unfairness_test_results, compress=9)
            joblib.dump(acc_test, accuracy_test_results, compress=9)

"""
def test_adult_local(data_prefix,test_prefix,exp_prefix,rho, beta, k):
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
                    #if np.isnan(adult_mdl.rule_description_[k][i]):
                        #continue
                    #raw_rule = adult_mdl.rule_description_[k][i]
                    #print(">>>>>>>>>>>"*5, raw_rule)
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
        ff = FairnessEvaluator(dataset["gender:Female"], dataset["gender:Male"], dataset["score"])
        unfairness_train.append(ff.demographic_parity_discrimination())
    # savaing unfairness and accurracy values on training set
    joblib.dump(unfairness_train, unfairness_train_results, compress=9)
    joblib.dump(acc_train, accuracy_train_results, compress=9)

"""