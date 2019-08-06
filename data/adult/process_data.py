import os
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost		

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle


from sklearn.externals import joblib
from scipy import stats


import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

RANDOM_STATE = 42

def stratify(dataset, attr, decision):

	strats = []
	for index, row in dataset.iterrows():
	    if((row[attr]==1) & (row[decision]==1)):
	        strats.append(0)

	    if((row[attr]==1) & (row[decision]==0)):
	        strats.append(1)

	    if((row[attr]==0) & (row[decision]==1)):
	        strats.append(2)

	    if((row[attr]==0) & (row[decision]==0)):
	        strats.append(3)

	dataset["strats"] = strats

	return dataset

def prepare(original_dataset_path):

    dataset = pd.read_csv(original_dataset_path)

    def basic_clean():
        dataset["gender"] = dataset["gender"].map({"Male": 1, "Female":0})
        dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
        dataset.drop(labels=["fnlwgt", "educational_num", "race", "native_country", "relationship"], axis = 1, inplace = True)    

    def process_education():
        education = []
        for index, row in dataset.iterrows():

            if(row['education'] in ["10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th", "Preschool"] ):
                education.append("dropout")

            if(row['education'] in ["Assoc-acdm", "Assoc-voc"] ):
                education.append("associates")

            if(row['education'] in ["Bachelors"] ):
                education.append("bachelors")

            if(row['education'] in ["Masters", "Doctorate"] ):
                education.append("masters_doctorate")

            if(row['education'] in ["HS-grad", "Some-college"] ):
                education.append("hs_grad")

            if(row['education'] in ["Prof-school"] ):
                education.append("prof_school")
            
        dataset['education'] = education

    def process_workclass():
        workclass = []
        for index, row in dataset.iterrows():

            if(row['workclass'] in ["Federal-gov"] ):
                workclass.append("fedGov")

            if(row['workclass'] in ["Local-gov", "State-gov"] ):
                workclass.append("otherGov")

            if(row['workclass'] in ["Private"] ):
                workclass.append("private")

            if(row['workclass'] in ["Self-emp-inc", "Self-emp-not-inc"] ):
                workclass.append("selfEmployed")

            if(row['workclass'] in ["Without-pay", "Never-worked" ] ):
                workclass.append("unEmployed")

        dataset['workclass'] = workclass

    def process_occupation():
        occupation = []
        for index, row in dataset.iterrows():

            if(row['occupation'] in ["Craft-repair", "Farming-fishing","Handlers-cleaners", "Machine-op-inspct", "Transport-moving"] ):
                occupation.append("blueCollar")

            if(row['occupation'] in ["Exec-managerial"] ):
                occupation.append("whiteCollar")
            
            if(row['occupation'] in ["Sales"] ):
                occupation.append("sales")

            if(row['occupation'] in ["Prof-specialty"] ):
                occupation.append("professional")

            if(row['occupation'] in ["Tech-support", "Protective-serv", "Armed-Forces", "Other-service", "Priv-house-serv", "Adm-clerical"] ):
                    occupation.append("other")

        dataset['occupation'] = occupation

    def process_marital():
        marital = []
        for index, row in dataset.iterrows():

            if(row['marital_status'] in ["Never-married"] ):
                marital.append("single")

            if(row['marital_status'] in ["Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse" ] ):
                marital.append("married")


        dataset['marital_status'] = marital

    def process_numerical():
        cols_to_scale_standard = ['hours_per_week', 'age']
        standardSc = StandardScaler()
        standardSc.fit(dataset[cols_to_scale_standard])

        cols_to_scale_robust = ['capital_gain', 'capital_loss']
        robustSc = RobustScaler()
        robustSc.fit(dataset[cols_to_scale_robust])

        def three_part(x, low, high):
            out = ""
            if (x < low ):
                out = "Low"

            if (x >= low and x <= high):
                out = "Normal"

            if (x > high ):
                out = "High"

            return out
        
        #for num in cols_to_scale_standard:
            #dataset[num] = dataset[num].apply( three_part, low=np.percentile(dataset[num], 25), high=np.percentile(dataset[num], 75) ) 

    basic_clean()
    process_education()
    process_workclass()
    process_occupation()
    process_marital()
    process_numerical()

    return dataset

#Random forest as black-box model
def build_and_predict_with_RF(dataset):

    df_train, df_bbl_test = train_test_split(dataset, test_size=0.66, stratify=dataset['income'], random_state = RANDOM_STATE)

    print("-------train file",len(df_train))

    df_test, df_bbl = train_test_split(df_bbl_test, test_size=0.5, stratify=df_bbl_test['income'], random_state = RANDOM_STATE)

    print("-------bbl file",len(df_bbl))

    print("-------test file",len(df_test))

    def train(df_train, df_bbl, df_test):
        # prepare bbClf training data
        y_train = df_train.income.values
        df_train.drop(labels=["income"], axis = 1, inplace = True)
        X_train = df_train
        X_train = pd.get_dummies(X_train)

        # prepare bbClf labelled data
        y_bbl = df_bbl.income.values
        df_bbl.drop(labels=["income"], axis = 1, inplace = True)
        X_bbl = df_bbl
        X_bbl = pd.get_dummies(X_bbl)

        # prepare bbClf test data
        y_test = df_test.income.values
        df_test.drop(labels=["income"], axis = 1, inplace = True)
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

        #joblib.dump(random_forest, "./pretrain/_rfModel.dump", compress=9)

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
        df_bbl['income'] = predictions_bbl.tolist()
        df_bbl.to_csv("./processed/_labelled.csv", encoding='utf-8', index=False)
        # rebuilding the training dataset
        df_train = pd.DataFrame(X_train, columns=columns)
        df_train['income'] = y_train
        df_train.to_csv("./processed/_train.csv", encoding='utf-8', index=False)
        # rebuilding the test dataset
        df_test = pd.DataFrame(X_test, columns=columns)
        df_test['income'] = y_test
        df_test.to_csv("./processed/_test.csv", encoding='utf-8', index=False)

    train(df_train, df_bbl, df_test)

#SVM as black-box model
def build_and_predict_with_SVM(dataset):

    df_train, df_bbl_test = train_test_split(dataset, test_size=0.66, stratify=dataset['income'], random_state = RANDOM_STATE)
    print("-------train file",len(df_train))
    df_test, df_bbl = train_test_split(df_bbl_test, test_size=0.5, stratify=df_bbl_test['income'], random_state = RANDOM_STATE)
    print("-------bbl file",len(df_bbl))
    print("-------test file",len(df_test))

    def train(df_train, df_bbl, df_test):
        # prepare bbClf training data
        y_train = df_train.income.values
        df_train.drop(labels=["income"], axis = 1, inplace = True)
        X_train = df_train
        X_train = pd.get_dummies(X_train)

        # prepare bbClf labelled data
        y_bbl = df_bbl.income.values
        df_bbl.drop(labels=["income"], axis = 1, inplace = True)
        X_bbl = df_bbl
        X_bbl = pd.get_dummies(X_bbl)

        # prepare bbClf test data
        y_test = df_test.income.values
        df_test.drop(labels=["income"], axis = 1, inplace = True)
        X_test = df_test
        X_test = pd.get_dummies(X_test)

        mdl = SVC(probability = True, random_state = RANDOM_STATE)
        #auc = make_scorer(roc_auc_score)
        rand_list = {"C": stats.uniform(1, 10), "gamma": stats.uniform(0.1, 1)}

        #, scoring = auc
        svm_random = RandomizedSearchCV(mdl, param_distributions = rand_list, n_iter = 3, n_jobs = -1, cv = 100, verbose=2, random_state = 42) 
        svm_random.fit(X_train, y_train)

        svc = svm_random.best_estimator_
        # getting the best model

        #joblib.dump(svc, "./pretrain/_svmModel.dump", compress=9)

        #getting prediction on the training set
        predictions_train = svc.predict(X_train)

        accuracy_train  = 100*accuracy_score(y_train, predictions_train)
        precision_train = 100*precision_score(y_train, predictions_train)
        recall_train    = 100*recall_score(y_train, predictions_train)

        print("===="*20)

        print("Accuracy train: %s%%" % accuracy_train)
        print("precision train: %s%%" % precision_train)
        print("recall train: %s%%" % recall_train)

        print("===="*20)

        #getting prediction on the labelled set
        predictions_bbl = svc.predict(X_bbl)

        accuracy_bbl  = 100*accuracy_score(y_bbl, predictions_bbl)
        precision_bbl = 100*precision_score(y_bbl, predictions_bbl)
        recall_bbl    = 100*recall_score(y_bbl, predictions_bbl)

        print("Accuracy bbl: %s%%" % accuracy_bbl)
        print("precision bbl: %s%%" % precision_bbl)
        print("recall bbl: %s%%" % recall_bbl)
        
        print("===="*20)

        #getting prediction on the test set
        predictions_test = svc.predict(X_test)

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
        df_bbl['income'] = predictions_bbl.tolist()
        df_bbl.to_csv("./processed/_labelled.csv", encoding='utf-8', index=False)
        # rebuilding the training dataset
        df_train = pd.DataFrame(X_train, columns=columns)
        df_train['income'] = y_train
        df_train.to_csv("./processed/_train.csv", encoding='utf-8', index=False)
        # rebuilding the test dataset
        df_test = pd.DataFrame(X_test, columns=columns)
        df_test['income'] = y_test
        df_test.to_csv("./processed/_test.csv", encoding='utf-8', index=False)

    train(df_train, df_bbl, df_test)

#DNN as black-box model
def build_and_predict_with_DNN(dataset):

    df_train, df_bbl_test = train_test_split(dataset, test_size=0.66, stratify=dataset['income'], random_state = RANDOM_STATE)

    print("-------train file",len(df_train))

    df_test, df_bbl = train_test_split(df_bbl_test, test_size=0.5, stratify=df_bbl_test['income'], random_state = RANDOM_STATE)

    print("-------bbl file",len(df_bbl))

    print("-------test file",len(df_test))

    def train(df_train, df_bbl, df_test):
        # prepare bbClf training data
        y_train = df_train.income.values
        df_train.drop(labels=["income"], axis = 1, inplace = True)
        X_train = df_train
        X_train = pd.get_dummies(X_train)

        # prepare bbClf labelled data
        y_bbl = df_bbl.income.values
        df_bbl.drop(labels=["income"], axis = 1, inplace = True)
        X_bbl = df_bbl
        X_bbl = pd.get_dummies(X_bbl)

        # prepare bbClf test data
        y_test = df_test.income.values
        df_test.drop(labels=["income"], axis = 1, inplace = True)
        X_test = df_test
        X_test = pd.get_dummies(X_test)

        mdl = make_pipeline(
            MLPClassifier(
                        solver='adam',
                        alpha=0.0001,
                        activation='relu',
                        batch_size=150,
                        hidden_layer_sizes=(200, 100),
                        random_state=1)
        )

        mdl.fit(X_train, y_train)

        #joblib.dump(mdl, "./pretrain/_dnnModel.dump", compress=9)

        #getting prediction on the training set
        predictions_train = mdl.predict(X_train)

        accuracy_train  = 100*accuracy_score(y_train, predictions_train)
        precision_train = 100*precision_score(y_train, predictions_train)
        recall_train    = 100*recall_score(y_train, predictions_train)

        print("===="*20)

        print("Accuracy train: %s%%" % accuracy_train)
        print("precision train: %s%%" % precision_train)
        print("recall train: %s%%" % recall_train)

        print("===="*20)

        #getting prediction on the labelled set
        predictions_bbl = mdl.predict(X_bbl)

        accuracy_bbl  = 100*accuracy_score(y_bbl, predictions_bbl)
        precision_bbl = 100*precision_score(y_bbl, predictions_bbl)
        recall_bbl    = 100*recall_score(y_bbl, predictions_bbl)

        print("Accuracy bbl: %s%%" % accuracy_bbl)
        print("precision bbl: %s%%" % precision_bbl)
        print("recall bbl: %s%%" % recall_bbl)
        
        print("===="*20)

        #getting prediction on the test set
        predictions_test = mdl.predict(X_test)

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
        df_bbl['income'] = predictions_bbl.tolist()
        df_bbl.to_csv("./processed/_labelled.csv", encoding='utf-8', index=False)
        # rebuilding the training dataset
        df_train = pd.DataFrame(X_train, columns=columns)
        df_train['income'] = y_train
        df_train.to_csv("./processed/_train.csv", encoding='utf-8', index=False)
        # rebuilding the test dataset
        df_test = pd.DataFrame(X_test, columns=columns)
        df_test['income'] = y_test
        df_test.to_csv("./processed/_test.csv", encoding='utf-8', index=False)

    train(df_train, df_bbl, df_test)

#XGBOOST as black-box model
def build_and_predict_with_XGBOOST(dataset):

    df_train, df_bbl_test = train_test_split(dataset, test_size=0.66, stratify=dataset['income'], random_state = RANDOM_STATE)

    print("-------train file",len(df_train))

    df_test, df_bbl = train_test_split(df_bbl_test, test_size=0.5, stratify=df_bbl_test['income'], random_state = RANDOM_STATE)

    print("-------bbl file",len(df_bbl))

    print("-------test file",len(df_test))

    def train(df_train, df_bbl, df_test):
        # prepare bbClf training data
        y_train = df_train.income.values
        df_train.drop(labels=["income"], axis = 1, inplace = True)
        X_train = df_train
        X_train = pd.get_dummies(X_train)

        # prepare bbClf labelled data
        y_bbl = df_bbl.income.values
        df_bbl.drop(labels=["income"], axis = 1, inplace = True)
        X_bbl = df_bbl
        X_bbl = pd.get_dummies(X_bbl)

        # prepare bbClf test data
        y_test = df_test.income.values
        df_test.drop(labels=["income"], axis = 1, inplace = True)
        X_test = df_test
        X_test = pd.get_dummies(X_test)

        params = {
        'xgbclassifier__gamma': [0.5, 1],
        'xgbclassifier__max_depth': [3, 4]
        }

        mdl = make_pipeline(
           xgboost.XGBClassifier(
                          n_estimators=600,
                          objective='binary:logistic',
                          silent=True,
                          nthread=1)
        )

        xgb_random = RandomizedSearchCV(mdl, param_distributions = params, n_iter = 4, n_jobs = -1, cv = 5, verbose=2, random_state = 42, scoring='accuracy') 
        xgb_random.fit(X_train, y_train)
        mdl = xgb_random.best_estimator_

        #joblib.dump(mdl, "./pretrain/_xgbModel.dump", compress=9)

        #getting prediction on the training set
        predictions_train = mdl.predict(X_train)

        accuracy_train  = 100*accuracy_score(y_train, predictions_train)
        precision_train = 100*precision_score(y_train, predictions_train)
        recall_train    = 100*recall_score(y_train, predictions_train)

        print("===="*20)

        print("Accuracy train: %s%%" % accuracy_train)
        print("precision train: %s%%" % precision_train)
        print("recall train: %s%%" % recall_train)

        print("===="*20)

        #getting prediction on the labelled set
        predictions_bbl = mdl.predict(X_bbl)

        accuracy_bbl  = 100*accuracy_score(y_bbl, predictions_bbl)
        precision_bbl = 100*precision_score(y_bbl, predictions_bbl)
        recall_bbl    = 100*recall_score(y_bbl, predictions_bbl)

        print("Accuracy bbl: %s%%" % accuracy_bbl)
        print("precision bbl: %s%%" % precision_bbl)
        print("recall bbl: %s%%" % recall_bbl)
        
        print("===="*20)

        #getting prediction on the test set
        predictions_test = mdl.predict(X_test)

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
        df_bbl['income'] = predictions_bbl.tolist()
        df_bbl.to_csv("./processed/_labelled.csv", encoding='utf-8', index=False)
        # rebuilding the training dataset
        df_train = pd.DataFrame(X_train, columns=columns)
        df_train['income'] = y_train
        df_train.to_csv("./processed/_train.csv", encoding='utf-8', index=False)
        # rebuilding the test dataset
        df_test = pd.DataFrame(X_test, columns=columns)
        df_test['income'] = y_test
        df_test.to_csv("./processed/_test.csv", encoding='utf-8', index=False)

    train(df_train, df_bbl, df_test)

# generic black-box model
def build_and_predict_with_bbox(dataset, model):

    df_train, df_bbl_test = train_test_split(dataset, test_size=0.66, stratify=dataset['income'], random_state = RANDOM_STATE)

    print("-------train file",len(df_train))

    df_test, df_bbl = train_test_split(df_bbl_test, test_size=0.5, stratify=df_bbl_test['income'], random_state = RANDOM_STATE)

    print("-------bbl file",len(df_bbl))

    print("-------test file",len(df_test))

    def train(df_train, df_bbl, df_test):
        # prepare bbClf training data
        y_train = df_train.income.values
        df_train.drop(labels=["income"], axis = 1, inplace = True)
        X_train = df_train
        X_train = pd.get_dummies(X_train)

        # prepare bbClf labelled data
        y_bbl = df_bbl.income.values
        df_bbl.drop(labels=["income"], axis = 1, inplace = True)
        X_bbl = df_bbl
        X_bbl = pd.get_dummies(X_bbl)

        # prepare bbClf test data
        y_test = df_test.income.values
        df_test.drop(labels=["income"], axis = 1, inplace = True)
        X_test = df_test
        X_test = pd.get_dummies(X_test)

        model_path = "./pretrain/" + model
        mdl = joblib.load(model_path)
        
        predictions_train = mdl.predict(X_train)

        accuracy_train  = 100*accuracy_score(y_train, predictions_train)
        precision_train = 100*precision_score(y_train, predictions_train)
        recall_train    = 100*recall_score(y_train, predictions_train)

        print("===="*20)

        print("Accuracy train: %s%%" % accuracy_train)
        print("precision train: %s%%" % precision_train)
        print("recall train: %s%%" % recall_train)

        print("===="*20)

        #getting prediction on the labelled set
        predictions_bbl = mdl.predict(X_bbl)

        accuracy_bbl  = 100*accuracy_score(y_bbl, predictions_bbl)
        precision_bbl = 100*precision_score(y_bbl, predictions_bbl)
        recall_bbl    = 100*recall_score(y_bbl, predictions_bbl)

        print("Accuracy bbl: %s%%" % accuracy_bbl)
        print("precision bbl: %s%%" % precision_bbl)
        print("recall bbl: %s%%" % recall_bbl)
        
        print("===="*20)

        #getting prediction on the test set
        predictions_test = mdl.predict(X_test)

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
        df_bbl['income'] = predictions_bbl.tolist()
        df_bbl['truth'] = y_bbl.tolist()
        df_bbl.to_csv("./processed/_labelled.csv", encoding='utf-8', index=False)
        # rebuilding the training dataset
        df_train = pd.DataFrame(X_train, columns=columns)
        df_train['income'] = y_train
        df_train.to_csv("./processed/_train.csv", encoding='utf-8', index=False)
        # rebuilding the test dataset
        df_test = pd.DataFrame(X_test, columns=columns)
        df_test['income'] = y_test
        df_test.to_csv("./processed/_test.csv", encoding='utf-8', index=False)

    train(df_train, df_bbl, df_test)

# binarization
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
    scores['income'] = df['income']
    features = df_bin

    data_bin=df_bin.values
    data_bin=data_bin.astype(int)
    df=df.reset_index(drop=True)
        
    #Getting labels
    Labels=["<=50K", ">50K"]
    label_bin=np.zeros((2,len(data_bin)))

    label_bin[0,:]=((df['income']==0).apply(int)).values
    label_bin[1,:]=((df['income']==1).apply(int)).values
    
    label_bin=label_bin.T
    label_bin=label_bin.astype(int)

    return df_bin, data_bin, label_bin, Labels, features, scores

# creating rules
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

    build_and_predict_with_RF(dataset)
    
    print("#"*15)
    dataset_labelled = pd.read_csv("./processed/_labelled.csv")
    df_bin, data_bin, label_bin, Labels, features, scores = binarized(dataset_labelled)
    create_rules_train("processed", df_bin, data_bin, label_bin, Labels, features, scores)

    dataset_test = pd.read_csv("./processed/_test.csv")
    df_bin, data_bin, label_bin, Labels, features, scores = binarized(dataset_test)
    create_rules_test("processed", df_bin, data_bin, label_bin, Labels, features, scores)
    

def save_rules_with_pretrain(original_dataset_path, model):

    os.makedirs('./processed', exist_ok=True)

    dataset = prepare(original_dataset_path)
    build_and_predict_with_bbox(dataset, model)
    
    print("#"*15)
    dataset_labelled = pd.read_csv("./processed/_labelled.csv")
    df_bin, data_bin, label_bin, Labels, features, scores = binarized(dataset_labelled)
    create_rules_train("processed", df_bin, data_bin, label_bin, Labels, features, scores)

    dataset_test = pd.read_csv("./processed/_test.csv")
    df_bin, data_bin, label_bin, Labels, features, scores = binarized(dataset_test)
    create_rules_test("processed", df_bin, data_bin, label_bin, Labels, features, scores)





