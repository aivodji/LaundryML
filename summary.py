import numpy as np
import pandas as pd
from fairness_eval import FairnessEvaluator
from collections import Counter


#summary of Adult
print("==================== # ==================== Adult")
print("------ train")
adult_train_file = "algs/corels/data/adult/processed/_train.csv"
adult_train = pd.read_csv(adult_train_file)
adult_train['gender:Male']=(adult_train['gender']==1).apply(int)
adult_train['gender:Female']=(adult_train['gender']==0).apply(int)
unfairness_adult_train = FairnessEvaluator(adult_train["gender:Female"], adult_train['gender:Male'], adult_train["income"])
print("unfairness on the training set: {}".format(unfairness_adult_train.demographic_parity_discrimination()))

print("------ labelled")
adult_label_file = "algs/corels/data/adult/processed/_labelled.csv"
adult_label = pd.read_csv(adult_label_file)
#print("Adult", "----"*10)
#print("size of suing group: {}".format(len(adult)))
#print("distribution of decision: {}".format(Counter(adult.income)))
adult_label['gender:Male']=(adult_label['gender']==1).apply(int)
adult_label['gender:Female']=(adult_label['gender']==0).apply(int)
unfairness_adult_label = FairnessEvaluator(adult_label["gender:Female"], adult_label['gender:Male'], adult_label["income"])
print("unfairness on the suing group: {}".format(unfairness_adult_label.demographic_parity_discrimination()))


print("------ test")
adult_test_file = "algs/corels/data/adult/processed/_test.csv"
adult_test = pd.read_csv(adult_test_file)
adult_test['gender:Male']=(adult_test['gender']==1).apply(int)
adult_test['gender:Female']=(adult_test['gender']==0).apply(int)
unfairness_adult_test = FairnessEvaluator(adult_test["gender:Female"], adult_test['gender:Male'], adult_test["income"])
print("unfairness on the test set: {}".format(unfairness_adult_test.demographic_parity_discrimination()))



# summary of compas
print("==================== # ==================== COMPAS")

print("------ training")
compas_train_file = "algs/corels/data/compas/processed/_train.csv"
compas_train = pd.read_csv(compas_train_file)
#print("size of suing group: {}".format(len(compas_train)))
#print("distribution of decision: {}".format(Counter(compas_label.two_year_recid)))
unfairness_compas_train = FairnessEvaluator(compas_train["race_African-American"], compas_train["race_Caucasian"], compas_train["two_year_recid"])
print("unfairness on the train set: {}".format(unfairness_compas_train.demographic_parity_discrimination()))

print("------ labelled")
compas_label_file = "algs/corels/data/compas/processed/_labelled.csv"
compas_label = pd.read_csv(compas_label_file)
unfairness_compas_label = FairnessEvaluator(compas_label["race_African-American"], compas_label["race_Caucasian"], compas_label["two_year_recid"])
print("unfairness on the suing group: {}".format(unfairness_compas_label.demographic_parity_discrimination()))

print("------ test")
compas_test_file = "algs/corels/data/compas/processed/_test.csv"
compas_test = pd.read_csv(compas_test_file)
unfairness_compas_test = FairnessEvaluator(compas_test["race_African-American"], compas_test["race_Caucasian"], compas_test["two_year_recid"])
print("unfairness on the test set: {}".format(unfairness_compas_test.demographic_parity_discrimination()))
