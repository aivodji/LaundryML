#!/usr/bin/env python3
# coding=utf-8

import os
import numpy as np
from sklearn.externals import joblib
#import matplotlib.pyplot as plt
from scipy.stats import entropy


os.chdir('.')

from RuleModels import CORELS_Enumerator


def convert_file(original_data_file, original_label_file, new_data_file, new_itemset_file=None, new_itemset_name_file=None):
    records = []
    record_names = []
    labels = []
    label_names = []
    with open(original_data_file, 'r') as f:
        for line in f:
            record = line.split()
            records.append([int(r) for r in record[1:]])
            record_names.append(record[0][1:-1])
        f.close()
    with open(original_label_file, 'r') as f:
        for line in f:
            record = line.split()
            labels.append([int(r) for r in record[1:]])
            label_names.append(record[0][1:-1])
        f.close()
    unique_record_names = [name for name in record_names if len(name.split(','))==1]
    unique_record_index = [i for i, name in enumerate(record_names) if len(name.split(','))==1]
    records = np.array(records).T
    records = records[:, unique_record_index]
    labels = np.array(labels).T
    with open(new_data_file, 'w') as f:
        for i in range(records.shape[0]):
            idx = np.where(records[i, :])[0]
            for val in idx:
                f.write('%d ' % (val,))
            f.write('%d\n' % labels[i, 1])
        f.close()
    if new_itemset_file is None:
        return
    with open(new_itemset_file, 'w') as f:
        for name in record_names:
            idx = [i for i, n in enumerate(unique_record_names) if n in name]
            for j, val in enumerate(idx):
                f.write('%d' % (val,))
                if j < len(idx) - 1:
                    f.write(' ')
            f.write('\n')
        f.close()
    with open(new_itemset_name_file, 'w') as f:
        f.write('#label_name\n')
        f.write('%s\n' % (label_names[1],))
        f.write('#itemset_name\n')
        for name in record_names:
            f.write('%s\n' % (name,))
        f.close()

def gen_sub_itemset(train_file, itemset_file, itemset_name_file, subitemset_file, subitemset_name_file, alpha=0.5):
    itemsets = []
    with open(itemset_file, 'r') as f:
        for line in f:
            itemsets.append(set([int(r) for r in line.split()]))
        f.close()
    labels = []
    names = []
    with open(itemset_name_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            elif i == 1:
                labels.append(line.strip())
            elif i == 2:
                continue
            else:
                names.append(line.strip())
        f.close()
    count_pos = [0] * len(itemsets)
    count = [0] * len(itemsets)
    with open(train_file, 'r') as f:
        for line in f:
            record = [int(r) for r in line.split()]
            label = record[-1]
            record = set(record[:-1])
            for i, itemset in enumerate(itemsets):
                if itemset.issubset(record):
                    if label == 1:
                        count_pos[i] += 1
                    count[i] += 1
        f.close()
    idx = [i for i, c in enumerate(count_pos) if c >= alpha * count[i]]
    with open(subitemset_file, 'w') as f:
        for i in idx:
            for j, val in enumerate(itemsets[i]):
                f.write('%d' % (val,))
                if j < len(itemsets[i]) - 1:
                    f.write(' ')
            f.write('\n')
        f.close()
    with open(subitemset_name_file, 'w') as f:
        f.write('#label_name\n')
        f.write('%s\n' % (labels[0],))
        f.write('#itemset_name\n')
        for i in idx:
            f.write('%s\n' % (names[i],))
        f.close()

def genData(data_prefix, exp_prefix, test_prefix='', alpha=0.5):
    # input file names
    train_file_items            = './algs/corels/data/%s/processed/_train.feature'        % (data_prefix)
    train_file_labels           = './algs/corels/data/%s/processed/_train%s.label'        % (data_prefix, test_prefix)

    test_file_items             = './algs/corels/data/%s/processed/_test.feature'        % (data_prefix)
    test_file_labels            = './algs/corels/data/%s/processed/_test%s.label'         % (data_prefix, test_prefix)

    # output files names
    train_file                  = './res/%s/%s/%s_train.txt'                           % (data_prefix, exp_prefix, data_prefix)
    test_file                   = './res/%s/%s/%s_test.txt'                            % (data_prefix, exp_prefix, data_prefix)

    itemset_file                = './res/%s/%s/%s_itemset.txt'                         % (data_prefix, exp_prefix, data_prefix)
    itemset_name_file           = './res/%s/%s/%s_itemset_name.txt'                    % (data_prefix, exp_prefix, data_prefix)
    subitemset_file             = './res/%s/%s/%s_itemset_pos%02d.txt'          	   % (data_prefix, exp_prefix, data_prefix, (int(100 * alpha)))
    subitemset_name_file        = './res/%s/%s/%s_itemset_name_pos%02d.txt'			% (data_prefix, exp_prefix, data_prefix, (int(100 * alpha)))

    # covert files
    convert_file(train_file_items, train_file_labels, train_file, itemset_file, itemset_name_file)
    convert_file(test_file_items, test_file_labels, test_file)
    gen_sub_itemset(train_file, itemset_file, itemset_name_file, subitemset_file, subitemset_name_file, alpha=alpha)


def genData_local(data_prefix, exp_prefix, test_prefix='', alpha=0.5):
    # input file names
    train_file_items            = './algs/corels/data/%s/local/_train.feature'        % (data_prefix)
    train_file_labels           = './algs/corels/data/%s/local/_train%s.label'        % (data_prefix, test_prefix)


    # output files names
    train_file                  = './res_local/%s/%s/%s_train.txt'                           % (data_prefix, exp_prefix, data_prefix)

    itemset_file                = './res_local/%s/%s/%s_itemset.txt'                         % (data_prefix, exp_prefix, data_prefix)
    itemset_name_file           = './res_local/%s/%s/%s_itemset_name.txt'                    % (data_prefix, exp_prefix, data_prefix)
    subitemset_file             = './res_local/%s/%s/%s_itemset_pos%02d.txt'          	   % (data_prefix, exp_prefix, data_prefix, (int(100 * alpha)))
    subitemset_name_file        = './res_local/%s/%s/%s_itemset_name_pos%02d.txt'			% (data_prefix, exp_prefix, data_prefix, (int(100 * alpha)))

    # covert files
    convert_file(train_file_items, train_file_labels, train_file, itemset_file, itemset_name_file)
    gen_sub_itemset(train_file, itemset_file, itemset_name_file, subitemset_file, subitemset_name_file, alpha=alpha)

   
def test_corels(data_prefix, exp_prefix, test_prefix='', rho=0.015, beta=0.0, k=50, opt='-c 2 -p 1', branch_depth=np.inf):

    data_file               = './res/%s/%s/%s_train.txt'                           % (data_prefix, exp_prefix, data_prefix)
    test_file               = './res/%s/%s/%s_test.txt'                            % (data_prefix, exp_prefix, data_prefix)
    minor_file            = './algs/corels/data/%s/processed/_train.minor'        % (data_prefix)

    itemset_file            = './res/%s/%s/%s_itemset.txt'                         % (data_prefix, exp_prefix, data_prefix)
    itemset_name_file       = './res/%s/%s/%s_itemset_name.txt'                    % (data_prefix, exp_prefix, data_prefix)
    model_file              = './res/%s/%s/%s_corels.mdl'                     % (data_prefix, exp_prefix, data_prefix)
    res_file                = './res/%s/%s/%s_corels_result.txt'              % (data_prefix, exp_prefix, data_prefix)

    model = CORELS_Enumerator(k=k, opt=opt, rho=rho, beta=beta, branch_depth=branch_depth, minimal=False)
    model.fit(data_file, itemset_file, itemset_name_file, minor_file)
    joblib.dump(model, model_file, compress=9)
    z, acc = model.predict(test_file)
    res = np.array([list(range(1, len(model.rule_)+1)), [len(rule) for rule in model.rule_], model.obj_, acc]).T
    np.savetxt(res_file, res, delimiter=',')



def test_corels_local(data_prefix, exp_prefix, test_prefix='', rho=0.015, beta=0.0, k=50, opt='-c 2 -p 1', branch_depth=np.inf):

    data_file               = './res_local/%s/%s/%s_train.txt'                           % (data_prefix, exp_prefix, data_prefix)

    minor_file            = './algs/corels/data/%s/local/_train.minor'        % (data_prefix)


    itemset_file            = './res_local/%s/%s/%s_itemset.txt'                         % (data_prefix, exp_prefix, data_prefix)
    itemset_name_file       = './res_local/%s/%s/%s_itemset_name.txt'                    % (data_prefix, exp_prefix, data_prefix)
    model_file              = './res_local/%s/%s/%s_corels.mdl'                     % (data_prefix, exp_prefix, data_prefix)
    res_file                = './res_local/%s/%s/%s_corels_result.txt'              % (data_prefix, exp_prefix, data_prefix)

    model = CORELS_Enumerator(k=k, opt=opt, rho=rho, beta=beta, branch_depth=branch_depth, minimal=False)
    model.fit(data_file, itemset_file, itemset_name_file, minor_file)
    joblib.dump(model, model_file, compress=9)
    z, acc = model.predict(data_file)
    res = np.array([list(range(1, len(model.rule_)+1)), [len(rule) for rule in model.rule_], model.obj_, acc]).T
    np.savetxt(res_file, res, delimiter=',')