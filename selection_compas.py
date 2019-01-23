import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.externals import joblib
import numpy as np

from RuleList import *

       
def best_model(dataset, experiments, target):
    models = []
    distances = []
    
    for experiment in experiments:
        unfairness_results = './res/%s/%s/unfairness_test.dump' %(dataset, experiment)
        accuracy_results = './res/%s/%s/accuracy_test.dump' %(dataset, experiment)
    
        acc = joblib.load(accuracy_results)
        #diff_metric = joblib.load(diffMetric_results)
        unfairness = joblib.load(unfairness_results)
        for j in range(len(unfairness)):
            rl = RuleList(unfairness=unfairness[j], fidelity=acc[j], beta=experiment + '_' + str(j))
            dist = rl.distance_to_target(target)
            models.append(rl)
            distances.append(dist)
            

    idx = np.argmin(distances)
    best = models[idx]

    print(best)

    return best


def get_models(dataset, experiments, target):
    models = []

    for experiment in experiments:

        # perfornmance on train
        accuracy_train_results = './res/%s/%s/accuracy_train.dump' %(dataset, experiment)
        unfairness_train_results = './res/%s/%s/unfairness_train.dump' %(dataset, experiment)
    
        # perfornmance on test
        accuracy_test_results = './res/%s/%s/accuracy_test.dump' %(dataset, experiment)
        unfairness_test_results = './res/%s/%s/unfairness_test.dump' %(dataset, experiment)
    
        acc_train  = joblib.load(accuracy_train_results)
        unfairness_train = joblib.load(unfairness_train_results)

        acc_test  = joblib.load(accuracy_test_results)
        unfairness_test = joblib.load(unfairness_test_results)


        for j in range(len(unfairness_train)):
            rl = RuleList2(unfairness_train=unfairness_train[j], fidelity_train=acc_train[j], quality_train=1.0, unfairness_test=unfairness_test[j], fidelity_test=acc_test[j], quality_test=1.0, beta=experiment + '_' + str(j))
            dist, dist2 = rl.distance_to_target(target)
            models.append(rl)

    return models


dataset = "compas/original"

experiments = []

lambdas = [0.005, 0.01]
betas = [0.0, 0.1, 0.2, 0.5, 0.7, 0.9]
target = RuleList2(unfairness_train=0.0, fidelity_train=1.0, unfairness_test=0.0, fidelity_test=1.0, beta="reg=0.0")

for _lambdak in lambdas:
    for _beta in betas:
        folder = '/lambda='+str(_lambdak) + '__' + 'beta='+str(_beta)
        experiments.append(folder) 


models = get_models(dataset, experiments, target)

print("Top 10 models" ,"||>>>"*15)


ll = filter(lambda x: x.unfairness_train < 0.085, models)
ll= sorted(ll, key=lambda x: x.fidelity_train, reverse=True)[:10]


for ml in ll:
        print(ml)

