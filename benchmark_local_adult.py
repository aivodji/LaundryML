from algs.corels.data.adult.process_local import *
from _adult import *
from plot import *
from _audit import *
import numpy as np
from sklearn.externals.joblib import Parallel, delayed

import time

def compute_neighborhood(data_prefix):
    os.chdir('./algs/corels/data/%s' % data_prefix)
    n = int(14925*0.1)
    computeNearestNeighbors_female(n)
    os.chdir('../../../..')

def compute_unfairness(data_prefix):
    os.chdir('./algs/corels/data/%s' % data_prefix)
    computeUnfairness()
    os.chdir('../../../..')

def getUsers(data_prefix):
    os.chdir('./algs/corels/data/%s' % data_prefix)
    users = get_all_user()
    os.chdir('../../../..')
    return users
    

def create_rules(data_prefix, users, idx):
    os.chdir('./algs/corels/data/%s' % data_prefix)
    create_rules_idx(users, idx)
    os.chdir('../../../..')

def bench(experiment_name="original", data_prefix='adult', test_prefix='', k=50, beta=0.0, rho=0.0):
    exp_prefix= experiment_name + '/lambda='+str(rho) + '__' + 'beta='+str(beta) 
    os.makedirs('./res_local/%s/%s'  %(data_prefix, exp_prefix), exist_ok=True)
    test_adult_local(data_prefix, test_prefix, exp_prefix, rho, beta, k)


compute_neighborhood(data_prefix='adult')
compute_unfairness(data_prefix='adult')

users = getUsers(data_prefix='adult')
time.sleep(1)

lambdas = [0.005]
betas = [0.1, 0.3, 0.5, 0.7, 0.9]


frames = []
compt = 0
for user_id, user_neighbors in enumerate(users):
        compt += 1

        create_rules('adult', users,  user_id)
        time.sleep(.100)
        df = pd.DataFrame()
        print('Processing user ----------- ', user_id)
        for _lambdak in lambdas:
                Parallel(n_jobs=-1)(delayed(bench)(beta=_beta, rho=_lambdak) for _beta in betas)
        data_prefix='adult'
        experiment_name="original"
        for _lambdak in lambdas:
                for _beta in betas:
                        
                        exp_prefix= experiment_name + '/lambda='+str(_lambdak) + '__' + 'beta='+str(_beta) 
                        enumplot_local(data_prefix, exp_prefix) 
                        unfairness, fidelity = best_unfairness(data_prefix, exp_prefix)

                        df['unfairness_' + str(_beta)] = [unfairness]
                        df['fidelity_' + str(_beta)] = [fidelity]
        frames.append(df)

users_df = pd.concat(frames)

users_df.to_csv("./algs/corels/data/adult/local/badml_unfairness.csv", encoding='utf-8', index=False)

#save for plotting
users_df.to_csv("./plotting_scripts/data/badml_unfairness_adult.csv", encoding='utf-8', index=False)