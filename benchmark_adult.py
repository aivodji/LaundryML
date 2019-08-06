from data.adult.process_data import save_rules, save_rules_with_pretrain
from plot import enumplot
from laundryML import LaundryML
import numpy as np
from sklearn.externals.joblib import Parallel, delayed

import time
import os

# experiments paramaters
params = {
        'original_dataset_path' : './original/adult.csv',
        'experiment_name' : 'global',
        'data_prefix' : 'adult',
        'test_prefix' : '',
        'res_folder' : 'res_rf',
        'k' : 10,
        'opt' : '-c 2 -p 1',
        'rho' : 0.0,
        'beta' : 0.0, 
        'metric' : 1, 
        'maj_pos' : 1, 
        'min_pos' : 2,
        'sensitve_attr' : 'gender:Female',
        'non_sensitve_attr' : 'gender:Male',
        'decision_attr' : 'income',
        'betas' :[0.0, 0.1, 0.2, 0.5, 0.7, 0.9],
        'metrics' : [1,2,3,4],
        'lambdas' : [0.005, 0.01],
        'pretrain' : True,
        'bbox' : '_rfModel.dump'
}

def create_rules(data_prefix, original_dataset_path):
    os.chdir('./data/%s' % data_prefix)
    save_rules(original_dataset_path)
    os.chdir('../..')

def create_rules_with_pretrain(data_prefix, original_dataset_path, model):
    os.chdir('./data/%s' % data_prefix)
    save_rules_with_pretrain(original_dataset_path, model)
    os.chdir('../..')

def bench(experiment_name=params['experiment_name'], data_prefix=params['data_prefix'], test_prefix=params['test_prefix'], res_folder=params['res_folder'], k=params['k'], opt=params['opt'], rho=params['rho'], beta=params['beta'], metric=params['metric'], maj_pos=params['maj_pos'], min_pos=params['min_pos'], sensitve_attr=params['sensitve_attr'], non_sensitve_attr=params['non_sensitve_attr'], decision_attr=params['decision_attr']):
    exp_prefix= experiment_name + '/lambda=' + str(rho) + '__' + 'beta='+ str(beta) + '__' + 'metric='+ str(metric)
    print("-"*30, exp_prefix)
    os.makedirs('./%s/%s/%s'  %(res_folder, data_prefix, exp_prefix), exist_ok=True)
    lml = LaundryML(data_prefix, test_prefix, exp_prefix, res_folder, k, opt, rho, beta, metric, maj_pos, min_pos, sensitve_attr, non_sensitve_attr, decision_attr)
    lml.run()

if params['pretrain']:
        create_rules_with_pretrain(data_prefix=params['data_prefix'], original_dataset_path=params['original_dataset_path'], model=params['bbox'])
else:
        create_rules(data_prefix=params['data_prefix'], original_dataset_path=params['original_dataset_path'])

time.sleep(35)


# searching for rationalization models
for _metric in params['metrics']:
        for _lambdak in params['lambdas']:
                Parallel(n_jobs=-1)(delayed(bench)(beta=_beta, rho=_lambdak, metric=_metric) for _beta in params['betas'])

# plotting results
for _metric in params['metrics']:
        for _lambdak in params['lambdas']:
                for _beta in params['betas']:
                        exp_prefix= params['experiment_name'] + '/lambda=' + str(_lambdak) + '__' + 'beta='+ str(_beta) + '__' + 'metric='+ str(_metric)
                        print("="*30, exp_prefix)
                        enumplot(params['res_folder'], params['data_prefix'], exp_prefix)
                        #get_audit(data_prefix, exp_prefix)
