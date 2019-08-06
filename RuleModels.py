import os
import codecs
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import RuleEnumerator as enumerator

class CORELS_Enumerator(object):
    def __init__(self, rho=0.001, beta=0.0, metric=1, maj_pos=1, min_pos=2, opt='-b', branch_depth=3, minimal=True, k=10):
        self.rho_ = rho
        self.opt_ = opt
        self.branch_depth_ = branch_depth
        self.minimal_ = minimal
        self.k_ = k
        self.beta_ = beta
        self.metric_ = metric
        self.maj_pos_ = maj_pos
        self.min_pos_ = min_pos
    
    def show(self, k):
        for i in range(len(self.pred_description_[k])):
            pred = self.pred_description_[k][i][1:-1]
            if i == 0:
                rule = self.rule_description_[k][i].replace('{', '').replace('}', '').split(',')
                print('IF ', end='')
                for j, r in enumerate(rule):
                    print(r, end='')
                    if j < len(rule) - 1:
                        print(' AND ', end='')
                print(' THEN ', end='')
                print(pred)
            elif i == len(self.pred_description_[k]) - 1:
                print('ELSE ', end='')
                print(pred)
            else:
                rule = self.rule_description_[k][i].replace('{', '').replace('}', '').split(',')
                print('ELSE IF ', end='')
                for j, r in enumerate(rule):
                    print(r, end='')
                    if j < len(rule) - 1:
                        print(' AND ', end='')
                print(' THEN ', end='')
                print(pred)
    
    def fit(self, data_file, itemset_file, itemset_name_file, minor_file):
        if self.minimal_:
            itemsets, data_name, label_name = enumerator.load_minimal_itemsets_and_names(itemset_file, itemset_name_file)
        else:
            itemsets, data_name, label_name = enumerator.load_itemsets_and_names(itemset_file, itemset_name_file)
        self.data_name_ = data_name
        self.label_name_ = label_name
        
        # corels
        #print('\t First CORELS...', end='')
        corels_data_file = data_file.replace('.txt', '_corels_data.txt')
        corels_label_file = data_file.replace('.txt', '_corels_label.txt')
        result_file = data_file.replace('.txt', '_corels_result.txt')
        enumerator.generate_corels_file(data_file, itemset_file, itemset_name_file, corels_data_file, corels_label_file, minimal=self.minimal_)
        # Call of CORELS
        model = CORELS(rho=self.rho_, beta=self.beta_, metric=self.metric_, maj_pos=self.maj_pos_, min_pos=self.min_pos_, opt=self.opt_, minimal=self.minimal_)
        model.fit(corels_data_file, corels_label_file, itemset_file, itemset_name_file, minor_file, result_file)
        supp = list(range(len(itemsets)))
        nonzeros = [(data_name.index(r), i) for i, r in enumerate(model.rule_description_[:-1])]
        removed = []
        nobranch = []
        searchlist = [(model.obj_, model.rule_, model.rule_description_, model.pred_, model.pred_decsription_, set(supp), set(nonzeros), set(removed), set(nobranch))]
        

        # enumeration
        self.obj_ = []
        self.rule_ = []
        self.rule_description_ = []
        self.pred_ = []
        self.pred_description_ = []
        count = 0
        while True:
            count += 1
            if len(searchlist) == 0:
                break
            i = np.argmax([v[0] for v in searchlist])
            obj = searchlist[i][0]
            rule = searchlist[i][1]
            rule_description = searchlist[i][2]
            pred = searchlist[i][3]
            pred_description = searchlist[i][4]
            supp = list(searchlist[i][5])
            nonzeros = list(searchlist[i][6])
            removed = list(searchlist[i][7])
            nobranch = list(searchlist[i][8])
            self.obj_.append(obj)
            self.rule_.append(rule)
            self.rule_description_.append(rule_description)
            self.pred_.append(pred)
            self.pred_description_.append(pred_description)
            searchlist.pop(i)
            if count >= self.k_:
                break
            #print('\t Search %3d (search size = %d):' % (count, len(set(nonzeros).difference(nobranch))), end='')
            t = 0
            
            for d in nonzeros:
                dep = d[1]
                d = d[0]
                if d in nobranch:
                    #print('*', [i for i, name in enumerate(data_name) if name in removed], d)
                    continue
                if dep >= self.branch_depth_:
                    continue

                
                supp.remove(d)
                removed.append(data_name[d])
                #print([i for i, name in enumerate(data_name) if name in removed])

                if len(supp) == 0:
                    continue
                t += 1
                #print('%d, ' % (t,), end='')
                
                # corels
                enumerator.generate_subitemset_file(itemset_file, itemset_name_file, removed=removed)
                subitemset_file = itemset_file.replace('.txt', '_sub.txt')
                subitemset_name_file = itemset_name_file.replace('.txt', '_sub.txt')
                enumerator.generate_corels_file(data_file, subitemset_file, subitemset_name_file, corels_data_file, corels_label_file, minimal=self.minimal_)
                
                # Call of CORELS
                model = CORELS(rho=self.rho_, beta=self.beta_, metric=self.metric_, maj_pos=self.maj_pos_, min_pos=self.min_pos_, opt=self.opt_, minimal=self.minimal_)
                
                model.fit(corels_data_file, corels_label_file, subitemset_file, subitemset_name_file, minor_file, result_file)
                nonzeros = [(data_name.index(r), i) for i, r in enumerate(model.rule_description_[:-1])]
                
                # post-check
                flg = False
                for v in searchlist:
                    if nonzeros == v[6]:
                        flg = True
                        break
                if flg:
                    supp.append(d)
                    removed.remove(data_name[d])
                    continue
                
                # update list
                #if (len(set(pred)) != 2):
                
                searchlist.append((model.obj_, model.rule_, model.rule_description_, model.pred_, model.pred_decsription_, set(supp), set(nonzeros), set(removed), set(nobranch)))
                supp.append(d)
                removed.remove(data_name[d])
                nobranch.append(d)
            #print('done.')
        
        # termination
        os.remove(corels_data_file)
        os.remove(corels_label_file)
        os.remove(result_file)
        #if self.k_ > 1:
        if len(self.pred_description_) > 1:
            os.remove(subitemset_file)
            os.remove(subitemset_name_file)
        
    def predict(self, data_file):
        y = []
        y_true = []
        with open(data_file, 'r') as f:
            for line in f:
                record = [int(item) for item in line.split()]
                data = set(record[:-1])
                y_true.append(record[-1])
                z = []
                for i, rule in enumerate((self.rule_)):
                    pred = self.pred_[i]
                    for j in range(len(rule)):
                        if rule[j].issubset(data):
                            z.append(pred[j])
                            break
                y.append(z)
        acc = [0] * len(y[0])
        for i in range(len(y)):
            for j in range(len(y[i])):
                acc[j] += (y[i][j] == y_true[i])
        acc = [a / len(y_true) for a in acc]
        return np.array(y), np.array(acc)

    def predict_local(self, data_file):
        lines = tuple(open(data_file, 'r'))
        line = lines[0]
        record = [int(item) for item in line.split()]
        data = set(record[:-1])
        y_true = record[-1]

        y = []
        
        for i, rule in enumerate((self.rule_)):
            pred = self.pred_[i]
            for j in range(len(rule)):
                if rule[j].issubset(data):
                    y.append(pred[j])
                    break
            
        acc = [0] * len(y)
        for i in range(len(y)):
            acc[i] = (y[i] == y_true)

        return np.array(acc)


class CORELS(object):
    def __init__(self, rho=0.001, beta=0.0, metric=1, maj_pos=1, min_pos=2, opt='-b', minimal=True):
        self.rho_ = rho
        self.beta_ = beta
        self.metric_ = metric
        self.maj_pos_ = maj_pos
        self.min_pos_ = min_pos

        self.opt_ = opt
        self.minimal_ = minimal
        self.obj_ = np.inf
        self.rule_ = []
        self.rule_description_ = []
        self.pred_ = []
        self.pred_decsription_ = []
        
    def fit(self, data_file, label_file, itemset_file, itemset_name_file, minor_file, result_file):
        os.system('./algs/corels/src/corels -n 100000 -r %f -z %f -w %d -x %d -y %d %s %s %s %s > %s' % (self.rho_, self.beta_, self.metric_,self.min_pos_, self.maj_pos_, self.opt_, data_file, label_file, minor_file, result_file))
        self.load_model(itemset_file, itemset_name_file, result_file)
    
    def load_model(self, itemset_file, itemset_name_file, result_file):
        if self.minimal_:
            itemsets, data_name, label_name = enumerator.load_minimal_itemsets_and_names(itemset_file, itemset_name_file)
        else:
            itemsets, data_name, label_name = enumerator.load_itemsets_and_names(itemset_file, itemset_name_file)
        with codecs.open(result_file, 'r', 'utf-8', 'ignore') as f:
            flg = False
            for line in f:
                if 'final min_objective' in line:
                    self.obj_ = 1 - float(line.split(':')[1].strip())
                if 'OPTIMAL RULE LIST' in line:
                    flg = True
                if flg:
                    if 'if (1) then' in line:
                        lines = line.split()
                        self.rule_description_.append(np.nan)
                        self.rule_.append(set([]))
                        self.pred_decsription_.append(lines[-1][1:-1])
                        self.pred_.append('True' in self.pred_decsription_[-1])
                    elif 'else if' in line:
                        lines = line.split()
                        self.rule_description_.append(lines[2][1:-1])
                        idx = np.where([(self.rule_description_[-1] == name) for name in data_name])[0]
                        self.rule_.append(itemsets[int(idx)])
                        self.pred_decsription_.append(lines[-1][1:-1])
                        self.pred_.append('True' in self.pred_decsription_[-1])
                    elif 'if' in line:
                        lines = line.split()
                        self.rule_description_.append(lines[1][1:-1])
                        idx = np.where([(self.rule_description_[-1] == name) for name in data_name])[0]
                        self.rule_.append(itemsets[int(idx)])
                        self.pred_decsription_.append(lines[-1][1:-1])
                        self.pred_.append('True' in self.pred_decsription_[-1])
                    elif 'else' in line:
                        lines = line.split()
                        self.rule_description_.append(np.nan)
                        self.rule_.append(set([]))
                        self.pred_decsription_.append(lines[-1][1:-1])
                        self.pred_.append('True' in self.pred_decsription_[-1])
        
    def predict(self, data_file):
        y = []
        acc = 0
        with open(data_file, 'r') as f:
            for line in f:
                record = [int(item) for item in line.split()]
                data = set(record[:-1])
                label = record[-1]
                for i in range(len(self.rule_)):
                    if self.rule_[i].issubset(data):
                        y.append(self.pred_[i])
                        break
                acc += (y[-1] == label)
        return np.array(y), acc / len(y)
