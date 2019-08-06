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

from collections import namedtuple



import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

RANDOM_STATE = 42


def computeUnfairness(dataset):
    dataset['minority']=(dataset['race_African-American'])
    dataset['majority']=(dataset['race_Caucasian'])
    ff = FairnessEvaluator(dataset['minority'], dataset['majority'], dataset['two_year_recid'])
    print(ff.demographic_parity_discrimination())

def prepare(original_dataset_path):

    dataset = pd.read_csv(original_dataset_path)
    
    def basic_clean():
        dataset["sex"] = dataset["sex"].map({"Male": 1, "Female":0})
        dataset.drop(labels=["sex-race", "age", "c_charge_desc"], axis = 1, inplace = True)    

    basic_clean()
    #process_numerical()

    #dataset.to_csv("./processed/_original.csv", encoding='utf-8', index=False)

    return dataset

def build_and_predict_with_bbClf(dataset):

    df_train, df_bbl_test = train_test_split(dataset, test_size=0.66, stratify=dataset['two_year_recid'], random_state = RANDOM_STATE)

    print("-------train file",len(df_train))

    df_test, df_bbl = train_test_split(df_bbl_test, test_size=0.5, stratify=df_bbl_test['two_year_recid'], random_state = RANDOM_STATE)

    print("-------bbl file",len(df_bbl))

    print("-------test file",len(df_test))

    def train(df_train, df_bbl, df_test):
        # prepare bbClf training data
        #df_bbl = pd.concat([df_bbl, df_train])
    
        y_train = df_train.two_year_recid.values
        df_train.drop(labels=["two_year_recid"], axis = 1, inplace = True)
        X_train = df_train
        X_train = pd.get_dummies(X_train)

        # prepare bbClf labelled data
        y_bbl = df_bbl.two_year_recid.values
        df_bbl.drop(labels=["two_year_recid"], axis = 1, inplace = True)
        X_bbl = df_bbl
        X_bbl = pd.get_dummies(X_bbl)

        # prepare bbClf test data
        y_test = df_test.two_year_recid.values
        df_test.drop(labels=["two_year_recid"], axis = 1, inplace = True)
        X_test = df_test
        X_test = pd.get_dummies(X_test)


        # training a random_forest
        rf = RandomForestClassifier(random_state = RANDOM_STATE)
        
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        # Use the random grid to search for best hyperparameters
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

        # Fit the random search model
        rf_random.fit(X_train, y_train)

        threshold = 0.5

        # getting the best model
        random_forest = rf_random.best_estimator_

        joblib.dump(random_forest, "./processed/_bestModel.dump", compress=9)

        #getting prediction on the training set
        predictions_proba_train = random_forest.predict_proba(X_train)
        predictions_train = (predictions_proba_train[:,1] >= threshold).astype('int')

        accuracy_train  = 100*accuracy_score(y_train, predictions_train)
        precision_train = 100*precision_score(y_train, predictions_train)
        recall_train    = 100*recall_score(y_train, predictions_train)

        print("===="*20)

        print("Accuracy train: %s%%" % accuracy_train)
        print("precision train: %s%%" % precision_train)
        print("recall train: %s%%" % recall_train)

        print("===="*20)

        #getting prediction on the labelled set
        predictions_proba_bbl = random_forest.predict_proba(X_bbl)
        predictions_bbl = (predictions_proba_bbl[:,1] >= threshold).astype('int')

        accuracy_bbl  = 100*accuracy_score(y_bbl, predictions_bbl)
        precision_bbl = 100*precision_score(y_bbl, predictions_bbl)
        recall_bbl    = 100*recall_score(y_bbl, predictions_bbl)

        print("Accuracy bbl: %s%%" % accuracy_bbl)
        print("precision bbl: %s%%" % precision_bbl)
        print("recall bbl: %s%%" % recall_bbl)
        
        print("===="*20)

        #getting prediction on the test set
        predictions_proba_test = random_forest.predict_proba(X_test)
        predictions_test = (predictions_proba_test[:,1] >= threshold).astype('int')

        accuracy_test  = 100*accuracy_score(y_test, predictions_test)
        precision_test = 100*precision_score(y_test, predictions_test)
        recall_test    = 100*recall_score(y_test, predictions_test)

        print("Accuracy test: %s%%" % accuracy_test)
        print("precision test: %s%%" % precision_test)
        print("recall test: %s%%" % recall_test)
        
        print("===="*20)


        
        columns = list(X_bbl)

        # rebuilding the bbl dataset
        df_bbl = pd.DataFrame(X_bbl, columns=columns)
        df_bbl['two_year_recid'] = predictions_bbl.tolist()
        df_bbl.to_csv("./processed/_labelled.csv", encoding='utf-8', index=False)
        # rebuilding the training dataset
        df_train = pd.DataFrame(X_train, columns=columns)
        df_train['two_year_recid'] = y_train
    
        df_train.to_csv("./processed/_train.csv", encoding='utf-8', index=False)
        # rebuilding the test dataset
        df_test = pd.DataFrame(X_test, columns=columns)
        df_test['two_year_recid'] = y_test
        
        df_test.to_csv("./processed/_test.csv", encoding='utf-8', index=False)

    train(df_train, df_bbl, df_test)

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
    scores['two_year_recid'] = df['two_year_recid']
    features = df_bin

    data_bin=df_bin.values
    data_bin=data_bin.astype(int)
    df=df.reset_index(drop=True)
        
    #Getting labels
    Labels=['recidivate-within-two-years:No','recidivate-within-two-years:Yes']
    label_bin=np.zeros((2,len(data_bin)))

    label_bin[0,:]=((df['two_year_recid']==0).apply(int)).values
    label_bin[1,:]=((df['two_year_recid']==1).apply(int)).values
    
    label_bin=label_bin.T
    label_bin=label_bin.astype(int)

    return df_bin, data_bin, label_bin, Labels, features, scores

def create_rules_train(folder, df_bin, data_bin, label_bin, Labels, features, scores):

    N = len(data_bin)

    #Pre processing minority label
    print('Création de la matrice minority label ...',end=' ')

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
        print(i, 'tous les éléments trouvés: ',len(index)==count[i])
        if len(lab_set)==2:
            for element in index:
                if (label_bin[element,0]!=majority):
                    minor_bin[element]=1
        print(len(unique_feat)-i,"uniques to explore\n")
   
    minor_bin=minor_bin.astype(int)
    print("   Done.\n")
    
    
    Rules=[]
    Rules=list(df_bin.columns)
        
    features.to_csv('processed/_auditing_train.csv', encoding='utf-8', index=False)
    scores.to_csv('processed/_scores_train.csv', encoding='utf-8', index=False)

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
        
    features.to_csv('processed/_auditing_test.csv', encoding='utf-8', index=False)
    scores.to_csv('processed/_scores_test.csv', encoding='utf-8', index=False)

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

def save_rules(original_dataset_path):

    os.makedirs('./processed', exist_ok=True)

    dataset = prepare(original_dataset_path)
    build_and_predict_with_bbClf(dataset)
   
    dataset_labelled = pd.read_csv("./processed/_labelled.csv")
    df_bin, data_bin, label_bin, Labels, features, scores = binarized(dataset_labelled)
    create_rules_train("processed", df_bin, data_bin, label_bin, Labels, features, scores)

    dataset_test = pd.read_csv("./processed/_test.csv")
    df_bin, data_bin, label_bin, Labels, features, scores = binarized(dataset_test)
    create_rules_test("processed", df_bin, data_bin, label_bin, Labels, features, scores)
 
 





