#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:27:36 2018
"""

import matplotlib.pyplot as plt
import os
from sklearn.externals import joblib
import numpy as np

def enumplot(prefix,exp_prefix):
    # experiments results urls
    objectiveFunction_results = './res/%s/%s/objectiveFunction.dump' %(prefix,exp_prefix)

    # files for train performances
    unfairness_train_results              = './res/%s/%s/unfairness_train.dump'        %  (prefix,exp_prefix)
    accuracy_train_results                = './res/%s/%s/accuracy_train.dump'           %(prefix,exp_prefix)

    # files for test performances
    unfairness_test_results              = './res/%s/%s/unfairness_test.dump'         %(prefix,exp_prefix)
    accuracy_test_results                = './res/%s/%s/accuracy_test.dump'           %(prefix,exp_prefix)

    img_file = './res/%s/%s/plot.png' %(prefix,exp_prefix)

    # objectives values
    obj_ = joblib.load(objectiveFunction_results)
    
    # unfairness and fidelity on training set
    unfairness_train = joblib.load(unfairness_train_results)
    fidelity_train = joblib.load(accuracy_train_results)

    # unfairness and fidelity on test set
    unfairness_test = joblib.load(unfairness_test_results)
    fidelity_test = joblib.load(accuracy_test_results)


    print(sorted(obj_, reverse=True) == obj_)

       
    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.plot(obj_,'o-b')
    plt.ylabel('Objective function')

    
    
    
    plt.subplot(2,2,2)
    plt.plot(fidelity_test,'o-g', label='test')
    plt.plot(fidelity_train,'-or', label='train')
    plt.legend(loc='upper right')
    plt.ylabel('Fidelity')
    
    plt.subplots_adjust(wspace=0.98)
    
    plt.subplot(2,2,3)
    plt.plot(unfairness_test,'o-g', label='test')
    plt.plot(unfairness_train,'-or', label='train')
    plt.legend(loc='upper right')
    plt.ylabel('Unfairness')


    """ plt.subplot(2,2,4)
    plt.plot(unfairness, acc, '+')
    plt.axis([0, 1, 0, 1])
    plt.ylabel('fidelity')
    plt.xlabel('unfairness') """

    

    fig.savefig(img_file)
    # plt.show()



def enumplot_local(prefix,exp_prefix):
    # experiments results urls
    objectiveFunction_results = './res_local/%s/%s/objectiveFunction.dump' %(prefix,exp_prefix)

    # files for train performances
    unfairness_train_results              = './res_local/%s/%s/unfairness_train.dump'        %  (prefix,exp_prefix)
    accuracy_train_results                = './res_local/%s/%s/accuracy_train.dump'           %(prefix,exp_prefix)


    img_file = './res_local/%s/%s/plot.png' %(prefix,exp_prefix)

    # objectives values
    obj_ = joblib.load(objectiveFunction_results)
    
    # unfairness and fidelity on training set
    unfairness_train = joblib.load(unfairness_train_results)
    fidelity_train = joblib.load(accuracy_train_results)

    


    #print(sorted(obj_, reverse=True) == obj_)

       
    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.plot(obj_,'o-b')
    plt.ylabel('Objective function')

    
    
    
    plt.subplot(2,2,2)
    plt.plot(fidelity_train,'-or', label='train')
    plt.ylabel('Fidelity')
    
    plt.subplots_adjust(wspace=0.98)
    
    plt.subplot(2,2,3)
    plt.plot(unfairness_train,'-or', label='train')
    plt.ylabel('Unfairness')


    """ plt.subplot(2,2,4)
    plt.plot(unfairness, acc, '+')
    plt.axis([0, 1, 0, 1])
    plt.ylabel('fidelity')
    plt.xlabel('unfairness') """

    

    fig.savefig(img_file)
    # plt.show()



def best_unfairness(prefix,exp_prefix):
    # files for train performances
    unfairness_train_results              = './res_local/%s/%s/unfairness_train.dump'        %  (prefix,exp_prefix)
    accuracy_train_results                = './res_local/%s/%s/accuracy_train.dump'           %(prefix,exp_prefix)

    # unfairness and fidelity on training set
    unfairness_train = joblib.load(unfairness_train_results)
    fidelity_train = joblib.load(accuracy_train_results)

    # print(unfairness_train)
    # print(fidelity_train)

    out = None

    all_val = [(unfairness_train[i], fidelity_train[i]) for i in range(len(unfairness_train))]

    all_val = sorted(all_val, key=lambda x: x[0], reverse=False)

    all_val_yes = [(x[0], x[1]) for x in all_val if x[1] ]

    #print("<<<<<<>>>>>>"*5, all_val_yes)

    if(len(all_val_yes) > 0):
        out = (all_val_yes[0][0], all_val_yes[0][1])
    else:
        out = (all_val[0][0], all_val[0][1])


    

    return out

    
