#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 09:13:17 2017
"""

import os

def run_ep(data_file, itemset_file, alpha=0.9, max_pat=10, min_sup=100, minimal=True):
    os.system('./algs/qtlamp/mht -of %s -wr %.3f -max_pat %d -min_sup %d E %s' % (itemset_file, alpha, max_pat, min_sup, data_file))
    if minimal:
        save_minimal_itemsets(itemset_file)

def load_data(data_file):
    y = []
    x = []
    with open(data_file, 'r') as f:
        for line in f:
            record = [int(item) for item in line.split()]
            x.append(set(record[:-1]))
            y.append(record[-1])
    return x, y

def load_minimal_itemsets(itemset_file):
    itemsets = []
    with open(itemset_file, 'r') as f:
        for i, line in enumerate(f):
            newset = set([int(item) for item in line.split()])
            if i == 0:
                itemsets.append(newset)
                continue
            flg = True
            redundant = []
            for j, itemset in enumerate(itemsets):
                if newset.issubset(itemset):
                    redundant.append(itemset)
                elif newset.issuperset(itemset):
                    flg = False
                    break
            if flg:
                for r in redundant:
                    itemsets.remove(r)
                itemsets.append(newset)
        f.close()
    return itemsets

def load_minimal_itemsets_and_names(itemset_file, itemset_name_file):
    itemsets = []
    data_name = []
    label_name = []
    with open(itemset_name_file, 'r') as g:
        flg = False
        for line in g:
            if '#label_name' in line:
                continue
            elif '#itemset_name' in line:
                flg = True
                break
            label_name.append(line.strip())
        with open(itemset_file, 'r') as f:
            for i, line in enumerate(f):
                newset = set([int(item) for item in line.split()])
                newname = g.readline().strip()
                if i == 0:
                    itemsets.append(newset)
                    data_name.append(newname)
                    continue
                flg = True
                redundant = []
                redundantname = []
                for j, itemset in enumerate(itemsets):
                    if newset.issubset(itemset):
                        redundant.append(itemset)
                        redundantname.append(data_name[j])
                    elif newset.issuperset(itemset):
                        flg = False
                        break
                if flg:
                    for r in redundant:
                        itemsets.remove(r)
                    for rn in redundantname:
                        data_name.remove(rn)
                    itemsets.append(newset)
                    data_name.append(newname)
            f.close()
        g.close()
    return itemsets, data_name, label_name

def load_itemsets_and_names(itemset_file, itemset_name_file):
    itemsets = []
    data_name = []
    label_name = []
    with open(itemset_file, 'r') as f:
        for line in f:
            itemset = set([int(r) for r in line.split()])
            itemsets.append(itemset)
        f.close()
    with open(itemset_name_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            elif i == 1:
                label_name.append(line.strip())
            elif i == 2:
                continue
            else:
                data_name.append(line.strip())
        f.close()
    return itemsets, data_name, label_name

def save_minimal_itemsets(itemset_file):
    itemsets = load_minimal_itemsets(itemset_file)
    with open(itemset_file, 'w') as f:
        for itemset in itemsets:
            for i, item in enumerate(itemset):
                f.write('%d' % (item,))
                if i < len(itemset) - 1:
                    f.write(' ')
            f.write('\n')
        f.close()
    
def merge_itemsets(itemset_file1, itemset_file2, merged_file):
    itemsets1 = load_minimal_itemsets(itemset_file1)
    itemsets2 = load_minimal_itemsets(itemset_file2)
    with open(merged_file, 'w') as f:
        for itemset in itemsets1:
            for i, item in enumerate(itemset):
                f.write('%d' % (item,))
                if i < len(itemset) - 1:
                    f.write(' ')
            f.write('\n')
        for itemset in itemsets2:
            for i, item in enumerate(itemset):
                f.write('%d' % (item,))
                if i < len(itemset) - 1:
                    f.write(' ')
            f.write('\n')
        f.close()

def generate_itemsetnames(name_file, itemset_file, itemset_name_file):
    
    # read name
    data_name = []
    label_name = []
    with open(name_file, 'r') as f:
        flg = False
        for line in f:
            if '#label_name' in line:
                continue
            elif '#itemset_name' in line:
                flg = True
                continue
            if flg:
                data_name.append(line.strip())
            else:
                label_name.append(line.strip())
    
    # write name
    with open(itemset_name_file, 'w') as f:
        f.write('#label_name\n')
        f.write('%s\n' % (label_name[0],))
        f.write('#itemset_name\n')
        with open(itemset_file, 'r') as g:
            for line in g:
                f.write('{')
                record = [int(val) for val in line.strip().split()]
                for i, val in enumerate(record):
                    f.write('%s' % (data_name[val],))
                    if i < len(record) - 1:
                        f.write(',')
                f.write('}\n')
            g.close()
        f.close()

"""
def load_itemset_name(itemset_name_file):
    data_name = []
    label_name = []
    with open(itemset_name_file, 'r') as f:
        flg = False
        for line in f:
            if '#label_name' in line:
                continue
            elif '#itemset_name' in line:
                flg = True
                continue
            if flg:
                data_name.append(line.strip())
            else:
                label_name.append(line.strip())
        f.close()
    return data_name, label_name
"""
    
def generate_corels_file(data_file, itemset_file, itemset_name_file, corels_data_file, corels_label_file, minimal=True):
    
    # read itemset
    #itemsets = load_minimal_itemsets(itemset_file)
    #data_name, label_name = load_itemset_name(itemset_name_file)
    if minimal:
        itemsets, data_name, label_name = load_minimal_itemsets_and_names(itemset_file, itemset_name_file)
    else:
        itemsets, data_name, label_name = load_itemsets_and_names(itemset_file, itemset_name_file)
    
    # read data
    data = []
    label = []
    with open(data_file, 'r') as f:
        for line in f:
            record = [int(item) for item in line.split()]
            label.append(record[-1])
            data.append([itemset.issubset(set(record[:-1])) for itemset in itemsets])
        f.close()
    
    # write to file
    with open(corels_data_file, 'w') as f:
        for i, itemset in enumerate(itemsets):
            f.write('%s ' % (data_name[i],))
            for j in range(len(data)):
                f.write('%d' % (data[j][i],))
                if j < len(data) - 1:
                    f.write(' ')
            f.write('\n')
        f.close()
    with open(corels_label_file, 'w') as f:
        f.write('{%s_is_False} ' % (label_name[0],))
        for j in range(len(label)):
            f.write('%d' % (1 - label[j],))
            if j < len(label) - 1:
                f.write(' ')
        f.write('\n')
        f.write('{%s_is_True} ' % (label_name[0],))
        for j in range(len(label)):
            f.write('%d' % (label[j],))
            if j < len(label) - 1:
                f.write(' ')
        f.write('\n')

def generate_dtree_file(data_file, index_file, name_file, tree_file):
    itemsets = load_minimal_itemsets(index_file)
    itemsets = set([list(itemset)[0] for itemset in itemsets])
    data_name, label_name = load_itemset_name(name_file)
    with open(tree_file, 'w') as f:
        with open(data_file, 'r') as g:
            for line in g:
                record = [int(item) for item in line.split()]
                label = record[-1]
                data = set(record[:-1])
                data = data.intersection(itemsets)
                for val in data:
                    f.write('%d ' % (val,))
                f.write('%d\n' % (label,))
            g.close()
        f.close()

def generate_subitemset_file(itemset_file, itemset_name_file, removed=[]):
    #itemsets = load_minimal_itemsets(itemset_file)
    #data_name, label_name = load_itemset_name(itemset_name_file)
    itemsets, data_name, label_name = load_minimal_itemsets_and_names(itemset_file, itemset_name_file)
    idx = [(name not in removed) for name in data_name]
    subitemset_file = itemset_file.replace('.txt', '_sub.txt')
    subitemset_name_file = itemset_name_file.replace('.txt', '_sub.txt')
    with open(subitemset_file, 'w') as f:
        for i, itemset in enumerate(itemsets):
            if not idx[i]:
                continue
            for j, val in enumerate(itemset):
                f.write('%d' % (val,))
                if j < len(itemset) - 1:
                    f.write(' ')
            f.write('\n')
        f.close()
    with open(subitemset_name_file, 'w') as f:
        f.write('#label_name\n')
        f.write('%s\n' % (label_name[0],))
        f.write('#itemset_name\n')
        for i, name in enumerate(data_name):
            if not idx[i]:
                continue
            f.write('%s\n' % (name,))
        f.close()
    
def generate_occ_file(data_file, itemset_file, occ_file):
    itemsets = []
    with open(itemset_file, 'r') as f:
        for line in f:
            itemset = set([int(r) for r in line.split()])
            itemsets.append(itemset)
        f.close()
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            record = [int(r) for r in line.split()]
            label = record[-1]
            record = set(record[:-1])
            if label == 1:
                data.append(record)
        f.close()
    with open(occ_file, 'w') as f:
        for i in range(len(itemsets)):
            f.write('1')
            if i < len(itemsets) - 1:
                f.write(' ')
        f.write('\n')
        for i in range(len(data)):
            f.write('1')
            if i < len(data) - 1:
                f.write(' ')
        f.write('\n')
        for itemset in itemsets:
            flg = False
            for i, record in enumerate(data):
                if itemset.issubset(record):
                    if flg:
                        f.write(' ')
                    f.write('%d' % (i,))
                    flg = True
            f.write('\n')
        f.close()
            
        