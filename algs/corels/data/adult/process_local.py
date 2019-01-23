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
    #Binarization of Adult features : Getting Rules
    df_bin = pd.DataFrame()

    #gender
    df_bin['gender:Male']=(df['gender']==1).apply(int)
    df_bin['gender:Female']=(df['gender']==0).apply(int)
    
    #age
    df_bin['age:<21']=(df.age <= 20.0).apply(int)
    df_bin['age:21-25']=( (df.age > 20.0) & (df.age <= 24.5)).apply(int)
    df_bin['age:26-28']=( (24.5 < df.age) & (df.age <= 27.5)).apply(int)
    df_bin['age:27-60']=( (27.5 < df.age) & (df.age <= 60.5)).apply(int)
    df_bin['age:>60']=(df.age > 60.5).apply(int)

    #Hours-per-week
    df_bin['hoursPerWeek:<=43']=(df.hours_per_week <= 42.5).apply(int)
    df_bin['hoursPerWeek:>43']=(df.hours_per_week > 42.5).apply(int)

    #capital_gain
    df_bin['capital_gain:<=7056']=(df.capital_gain <= 7055.5).apply(int)
    df_bin['capital_gain:>7056']=(df.capital_gain > 7055.5).apply(int)

    #education
    df_bin['education:dropout']=( df.education_dropout == 1 ).apply(int)
    df_bin['education:associates']=( df.education_associates == 1 ).apply(int)
    df_bin['education:bachelors']=( df.education_bachelors == 1 ).apply(int)
    df_bin['education:masters-doctorate']=( df.education_masters_doctorate == 1 ).apply(int)
    df_bin['education:HS-grad']=( df.education_hs_grad == 1 ).apply(int)
    df_bin['education:Prof-school']=( df.education_prof_school == 1 ).apply(int)


    
    #workclass
    df_bin['workclass:FedGov']=( df.workclass_fedGov ==1 ).apply(int)
    df_bin['workclass:OtherGov']=( df.workclass_otherGov ==1 ).apply(int)
    df_bin['workclass:private']=( df.workclass_private ==1 ).apply(int)
    df_bin['workclass:selfEmployed']=( df.workclass_selfEmployed ==1 ).apply(int)

    #Occupation
    df_bin['occupation:blue-collar']=(df.occupation_blueCollar == 1).apply(int)
    df_bin['occupation:white-collar']=(df.occupation_whiteCollar == 1).apply(int)
    df_bin['occupation:professional']=(df.occupation_professional == 1).apply(int)
    df_bin['occupation:sales']=(df.occupation_sales == 1).apply(int)
    df_bin['occupation:other']=(df.occupation_other == 1).apply(int)

    
    #marital status
    df_bin['marital:single']=(df.marital_status_single == 1).apply(int)
    df_bin['marital:married']= (df.marital_status_married == 1).apply(int)

    # end binarized

    scores = pd.DataFrame()
    scores['income'] = (df['income']==1).apply(int)
    features = df_bin

    data_bin=df_bin.values
    data_bin=data_bin.astype(int)
    df=df.reset_index(drop=True)
        
    #Getting labels
    Labels=[0,1]
    label_bin=np.zeros((2,len(data_bin)))

    label_bin[0,:]=((df['income']==0).apply(int)).values
    label_bin[1,:]=((df['income']==1).apply(int)).values
    
    label_bin=label_bin.T
    label_bin=label_bin.astype(int)

    return df_bin, data_bin, label_bin, Labels, features, scores

def create_rules_train(folder, df_bin, data_bin, label_bin, Labels, features, scores):

    N = len(data_bin)

    #Pre processing minority label
    #print('Création de la matrice minority label ...',end=' ')

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

def computeNearestNeighbors_female(nbrSamples):
    dataset_org = pd.read_csv("./processed/_labelled.csv")

    dataset = pd.read_csv("./processed/_labelled.csv")
    
    rejected_females = dataset_org.index[(dataset_org.gender == 0) & (dataset_org.income == 0)].tolist()

    all_user = []
    y = dataset.income.values

    dataset.drop(labels=["income"], axis = 1, inplace = True)
    X = dataset
    X = pd.get_dummies(X)

    neigh = NearestNeighbors(n_neighbors=nbrSamples)
    neigh.fit(X) 
    Z  = neigh.kneighbors(X, return_distance=False)

    compt = 0

    for idx  in rejected_females:
        
        selected = Z[idx]
        selected = [idx] + list(filter(lambda x: x != idx, selected))
        user_df = dataset_org.iloc[selected]


        ln = len(Counter(user_df.income))
        ln = ln > 1

        ff = FairnessEvaluator((user_df['gender']==0).apply(int), (user_df['gender']==1).apply(int), user_df['income'])
        unfairness = ff.demographic_parity_discrimination()

        unfairness = unfairness > 0.05

        if (ln & unfairness):
            print('processing user ', idx)
            all_user.append(user_df)
            compt += 1

        #if (compt == 200):
            #break

        
          
    joblib.dump(all_user, "./local/all_female.dump", compress=9)

def computeUnfairness():
    all_Users = joblib.load("./local/all_female.dump")
    all_unfairness = []

    for user_df in all_Users:
        ff = FairnessEvaluator((user_df['gender']==0).apply(int), (user_df['gender']==1).apply(int), user_df['income'])
        unfairness = ff.demographic_parity_discrimination()
        all_unfairness.append(unfairness)
    
    unfairness_df = pd.DataFrame()
    unfairness_df['unfairness'] = all_unfairness
    unfairness_df.to_csv("./local/bbox_unfairness.csv", encoding='utf-8', index=False)
    unfairness_df.to_csv("./local/bbox_unfairness.csv", encoding='utf-8', index=False)
    unfairness_df.to_csv("../../../../plotting_scripts/data/bbox_unfairness_adult.csv", encoding='utf-8', index=False)

def get_all_user():
    all_Users = joblib.load("./local/all_female.dump")
    return all_Users


def create_rules_idx(users, idx):
    user_df = users[idx]
    df_bin, data_bin, label_bin, Labels, features, scores = binarized(user_df)
    create_rules_train("local", df_bin, data_bin, label_bin, Labels, features, scores)







