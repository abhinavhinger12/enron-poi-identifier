#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
#import seaborn as sns

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments',
                                 'total_payments', 'loan_advances',
                                 'bonus', 'restricted_stock_deferred',
                                 'deferred_income', 'total_stock_value',
                                 'expenses', 'exercised_stock_options',
                                 'long_term_incentive', 'restricted_stock',
                                 'director_fees', 'from_poi_to_this_person',
                                 'from_this_person_to_poi',
                                 'shared_receipt_with_poi','to_messages',
                'from_messages'] # Initially we will add all features Later we will do Feature Selection to select the best features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Task 2: Remove outliers
data_dict.pop('TOTAL') #Only 1 outlier which is sum of all values of a feature which we dont want
### Task 3: Create new feature(s)
'''
We will create 2 new features that are
1. Fraction of received messages that are sent by poi (frac_from_poi)
2. Fraction of sent messages that are sent to poi (frac_to_poi)
'''
for key in my_dataset:

    if my_dataset[key]["from_poi_to_this_person"] == "NaN":  #from_messages and from_poi_to_this_person occur as NaN TOGETHER
        my_dataset[key]['frac_from_poi']=0
    else:
        frac = round(float(my_dataset[key]["from_poi_to_this_person"])/my_dataset[key]["from_messages"],3)
        my_dataset[key]['frac_from_poi']=frac

for key in my_dataset:

    if my_dataset[key]["from_this_person_to_poi"] == "NaN":  #from_messages and from_poi_to_this_person occur as NaN TOGETHER
        my_dataset[key]['frac_to_poi']=0
    else:
        frac = round(float(my_dataset[key]["from_this_person_to_poi"])/my_dataset[key]["from_messages"],3)
        my_dataset[key]['frac_to_poi']=frac
# for key in my_dataset:
#     print my_dataset[key]["frac_to_poi"] , "  " , my_dataset[key]["frac_to_poi"]
features_list = features_list + ['frac_from_poi','frac_to_poi']
# print my_dataset.keys()
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
'''
Now the complete list of Features is readyself.We want to Select the best features only to avoid overfitting (SelectKBest)
'''
k_best = SelectKBest()
k_best.fit(features, labels)

k_best = [features_list[i] for i in k_best.get_support(indices=True)]

print(' ')
print('k-best features:',k_best)
print(' ')
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
def scoreCLF(clf,features,labels,param_grid,cv,score):
    '''
    This function takes a classifier, features, target variable, paramter grid (for grid search)
    and fits and scores the resulting model.
    A confusion matrix and classification report is returned
    with the performance metrics.
    Takes inspiration from:  http://chrisstrelioff.ws/sandbox/2015/06/25/decision_trees_in_python_again_cross_validation.html
    And the know how to tune gridsearch from:
    https://www.kaggle.com/kevinarvai/fine-tuning-a-classifier-in-scikit-learn
    '''
    #importing relevant libraries
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report,confusion_matrix
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    skf = StratifiedKFold(n_splits=cv)

    # splitting the data
    features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

    if param_grid != None:
        #creating and initializing the grid search
        clf = GridSearchCV(clf, param_grid=param_grid, cv=skf,
                           scoring=score,
                           return_train_score=True, n_jobs=-1)
        clf.fit(features_train, labels_train)

        # printing best parameters
        print(' ')
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print(' ')
        bestP = clf.best_params_
    else:
        clf.fit(features_train, labels_train)
        bestP= []

    # getting predictions and confusion matrix
    pred = clf.predict(features_test)
    print(classification_report(labels_test, pred))
    print(' ')
    return bestP
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = AdaBoostClassifier()
params = [{'algorithm':['SAMME'],
         'n_estimators':[25,35,40,50,75,100],
             'learning_rate':[0.1,0.25,0.5,0.75,1]},
          {'algorithm':['SAMME.R'],
         'n_estimators':[25,35,40,50,75,100],
             'learning_rate':[0.1,0.25,0.5,0.75,1]}]
bestParams = scoreCLF(clf,features,labels,params,10,"recall")
clf = AdaBoostClassifier(**bestParams)
#clf = RandomForestClassifier(**bestParams)
#clf = LogisticRegression(**bestParams)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, k_best)
