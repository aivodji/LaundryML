import os
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle

from imblearn.over_sampling import SMOTE

from sklearn.externals import joblib


import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

RANDOM_STATE = 42

from sklearn.utils import check_random_state

from sklearn.neighbors import NearestNeighbors

from collections import namedtuple
import random


class FairnessEvaluator(namedtuple('FairnessEvaluator', 'minority majority label')):
    def demographic_parity_discrimination(self):
        yes_majority_prop = np.mean(np.logical_and(self.majority == 1, self.label == 1))
        yes_minority_prop = np.mean(np.logical_and(self.minority == 1, self.label == 1))

        dem_parity = abs(yes_majority_prop - yes_minority_prop)

        return dem_parity

def binarized(df):
    #Binarization 
    df_bin = pd.DataFrame()
    df_bin['race:Caucasian']=(df['race_Caucasian'])
    df_bin['race:AfricanAmerican']=(df['race_African-American'])
    df_bin['race:Asian']=(df['race_Asian'])
    df_bin['race:Hispanic']=(df['race_Hispanic'])
    df_bin['race:Other']=(df['race_Other'])
    df_bin['race:NativeAmerican']=(df['race_Native American'])

    #sex
    df_bin['gender:Male']=(df['sex']==1).apply(int)
    df_bin['gender:Female']=(df['sex']==0).apply(int)

    #age
    df_bin['age:25-45']=(df['age_cat_25 - 45'])
    df_bin['age:>45']=(df['age_cat_Greater than 45'])
    df_bin['age:<25']=(df['age_cat_Less than 25'])


    #degree_charge
    df_bin['charge:Felony']=(df['c_charge_degree_F'])
    df_bin['charge:Misdemeanor']=(df['c_charge_degree_M'])
    
    #juv_fel_count
    df_bin['juv_fel_count:<=1.5']=(df.juv_fel_count <= 1.5).apply(int)
    df_bin['juv_fel_count:>1.5']=(df.juv_fel_count > 1.5).apply(int)

    #juv_other_count
    df_bin['juv_other_count:<=0.5']=(df.juv_other_count <= 0.5).apply(int)
    df_bin['juv_other_count:>0.5']=(df.juv_other_count > 0.5).apply(int)

    #priors_count
    df_bin['priors_count:<1']=(df.priors_count <= 0.5).apply(int)
    df_bin['priors_count:0.5-1.5']=((0.5 < df.priors_count) & (df.priors_count <= 1.5)).apply(int)
    df_bin['priors_count:1.5-2.5']=((1.5 < df.priors_count) & (df.priors_count <= 2.5)).apply(int)
    df_bin['priors_count:2.5-4.5']=((2.5 < df.priors_count) & (df.priors_count <= 4.5)).apply(int)
    df_bin['priors_count:4.5-5.5']=((4.5 < df.priors_count) & (df.priors_count <= 5.5)).apply(int)
    df_bin['priors_count:5.5-8.5']=((5.5 < df.priors_count) & (df.priors_count <= 8.5)).apply(int)
    df_bin['priors_count:8.5-14.5']=((8.5 < df.priors_count) & (df.priors_count <= 14.5)).apply(int)
    df_bin['priors_count:14.5-15.5']=((14.5 < df.priors_count) & (df.priors_count <= 15.5)).apply(int)
    df_bin['priors_count:15.5-27.5']=((15.5 < df.priors_count) & (df.priors_count <= 27.5)).apply(int)
    df_bin['priors_count:>27.5']=(df.priors_count > 27.5).apply(int)

    
    # end binarized

    scores = pd.DataFrame()
    scores['two_year_recid'] = (df['two_year_recid']==0).apply(int)
    features = df_bin

    data_bin=df_bin.values
    data_bin=data_bin.astype(int)
    df=df.reset_index(drop=True)
        
    #Getting labels
    Labels=[0,1]
    label_bin=np.zeros((2,len(data_bin)))

    label_bin[1,:]=((df['two_year_recid']==1).apply(int)).values
    label_bin[0,:]=((df['two_year_recid']==0).apply(int)).values
    
    label_bin=label_bin.T
    label_bin=label_bin.astype(int)

    return df_bin, data_bin, label_bin, Labels, features, scores

def create_rules_train(folder, df_bin, data_bin, label_bin, Labels, features, scores):
    N = len(data_bin)
    
    unique_feat, indices, count = np.unique(data_bin[0:N,:], return_index=True, return_counts=True, axis=0)
    minor_bin=np.zeros(N)

    for i, element in enumerate(unique_feat):
        lab_set=set()
        lab_set.add(label_bin[indices[i],0])
        lab_list=[label_bin[indices[i],0]]
        index=set()
        index.add(indices[i])
        for j in range(indices[i]+1,N):
            if np.array_equal(element,data_bin[j,:]):
                index.add(j)
                lab_set.add(label_bin[j,0])
                lab_list.append(label_bin[j,0])
            if len(index)==count[i]:
                break
        majority=int(round(np.mean(lab_list)))
        #print(i, 'tous les éléments trouvés: ',len(index)==count[i])
        if len(lab_set)==2:
            for element in index:
                if (label_bin[element,0]!=majority):
                    minor_bin[element]=1
        #print(len(unique_feat)-i,"uniques to explore\n")
   
    minor_bin=minor_bin.astype(int)
    #print("   Done.\n")
    
    
    Rules=[]
    Rules=list(df_bin.columns)
        
    features.to_csv('local/_auditing_train.csv', encoding='utf-8', index=False)
    scores.to_csv('local/_scores_train.csv', encoding='utf-8', index=False)

    N = len(data_bin)
    
    # Ecriture fichier data train
    with open(folder + "/" + "_train.feature","w") as out:
        for j, item in enumerate(Rules):
            out.write("{")
            out.write(item)
            out.write("} ")
            for i in range (0,N):
                out.write(str(data_bin[i][j]))
                if i==N-1:
                    out.write("\n")
                else:
                    out.write(" ")
        out.close()
        
    
    #Ecriture fichier Label train biaisé
    with open(folder + "/"  + "_train.label","w") as out:
        for j, item in enumerate(Labels):
            out.write("{")
            out.write(str(item))
            out.write("} ")
            for i in range (0,N):
                out.write(str(label_bin[i][j]))
                if i==N-1:
                    out.write("\n")
                else:
                    out.write(" ")
        out.close()

    # Ecriture du fichier minor
    with open(folder + "/"  + "_train.minor","w") as minor:
        minor.write("{group_minority} ")
        for j in range(N):
            minor.write(str(minor_bin[j]))
            if j==N-1:
                minor.write("\n")
            else:
                minor.write(" ")
        minor.close()    

def create_rules_test(folder, df_bin, data_bin, label_bin, Labels, features, scores):
    
    Rules=[]
    Rules=list(df_bin.columns)
        
    features.to_csv('local/_auditing_test.csv', encoding='utf-8', index=False)
    scores.to_csv('local/_scores_test.csv', encoding='utf-8', index=False)

    N = len(data_bin)
    
    # Ecriture fichier data train
    with open(folder + "/" + "_test.feature","w") as out:
        for j, item in enumerate(Rules):
            out.write("{")
            out.write(item)
            out.write("} ")
            for i in range (0,N):
                out.write(str(data_bin[i][j]))
                if i==N-1:
                    out.write("\n")
                else:
                    out.write(" ")
        out.close()
        
    
    #Ecriture fichier Label train biaisé
    with open(folder + "/"  + "_test.label","w") as out:
        for j, item in enumerate(Labels):
            out.write("{")
            out.write(str(item))
            out.write("} ")
            for i in range (0,N):
                out.write(str(label_bin[i][j]))
                if i==N-1:
                    out.write("\n")
                else:
                    out.write(" ")
        out.close() 

def computeNearestNeighbors_minority(nbrSamples):
    dataset_org = pd.read_csv("./processed/_labelled.csv")

    dataset = pd.read_csv("./processed/_labelled.csv")
    
    rejected_minority = dataset_org.index[(dataset_org['race_African-American'] == 1) & (dataset_org.two_year_recid == 1)].tolist()

    all_user = []
    y = dataset.two_year_recid.values

    dataset.drop(labels=["two_year_recid"], axis = 1, inplace = True)
    X = dataset
    X = pd.get_dummies(X)

    neigh = NearestNeighbors(n_neighbors=nbrSamples)
    neigh.fit(X) 
    Z  = neigh.kneighbors(X, return_distance=False)

    compt = 0

    for idx  in rejected_minority:
        selected = Z[idx]
        selected = [idx] + list(filter(lambda x: x != idx, selected))
        user_df = dataset_org.iloc[selected]
        ln = len(Counter(user_df.two_year_recid))
        ln = ln > 1
        ff = FairnessEvaluator((user_df['race_African-American']==0).apply(int), (user_df['race_Caucasian']==1).apply(int), user_df['two_year_recid'])
        unfairness = ff.demographic_parity_discrimination()
        unfairness = unfairness > 0.05

        if (ln & unfairness):
            print('processing user ', idx, 'unfairness: ', unfairness)
            all_user.append(user_df)
            compt += 1


        
          
    joblib.dump(all_user, "./local/all_users.dump", compress=9)

def computeUnfairness():
    all_Users = joblib.load("./local/all_users.dump")
    all_unfairness = []

    for user_df in all_Users:
        ff = FairnessEvaluator((user_df['race_African-American']==0).apply(int), (user_df['race_Caucasian']==1).apply(int), user_df['two_year_recid'])
        unfairness = ff.demographic_parity_discrimination()
        all_unfairness.append(unfairness)
    
    unfairness_df = pd.DataFrame()
    unfairness_df['unfairness'] = all_unfairness
    unfairness_df.to_csv("./local/bbox_unfairness.csv", encoding='utf-8', index=False)
    unfairness_df.to_csv("../../../../plotting_scripts/data/bbox_unfairness_compas.csv", encoding='utf-8', index=False)


def get_all_user():
    all_Users = joblib.load("./local/all_users.dump")
    return all_Users

def create_rules_idx(users, idx):
    user_df = users[idx]
    df_bin, data_bin, label_bin, Labels, features, scores = binarized(user_df)
    create_rules_train("local", df_bin, data_bin, label_bin, Labels, features, scores)







