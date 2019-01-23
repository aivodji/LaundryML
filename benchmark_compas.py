from algs.corels.data.compas.process_data import *
from _compas import *
from plot import *
from _audit import *
import numpy as np
import time

from sklearn.externals.joblib import Parallel, delayed


def create_rules_once(data_prefix, original_dataset_path):
    os.chdir('./algs/corels/data/%s' % data_prefix)
    save_rules(original_dataset_path)
    os.chdir('../../../..')

create_rules_once(data_prefix='compas', original_dataset_path="./data/compas_clean.csv")
time.sleep(5)

def bench(experiment_name="original", data_prefix='compas', test_prefix='', k=50, beta=0.0, rho=0.0):
    exp_prefix= experiment_name + '/lambda='+str(rho) + '__' + 'beta='+str(beta) 
    print("-"*30, exp_prefix)
    os.makedirs('./res/%s/%s'  %(data_prefix, exp_prefix), exist_ok=True)
    test_compas(data_prefix, test_prefix, exp_prefix, rho, beta, k)
    

lambdas = [0.005, 0.01]
betas = [0.0, 0.1, 0.2, 0.5, 0.7, 0.9]

for _lambdak in lambdas:
    Parallel(n_jobs=-1)(delayed(bench)(beta=_beta, rho=_lambdak) for _beta in betas)


data_prefix='compas'
experiment_name="original"

for _lambdak in lambdas:
    for _beta in betas:
        exp_prefix= experiment_name + '/lambda='+str(_lambdak) + '__' + 'beta='+str(_beta) 
        enumplot(data_prefix, exp_prefix) 
        get_audit(data_prefix, exp_prefix)