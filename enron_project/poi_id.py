
# coding: utf-8

# # Identify Fraud from Enron Email
# ## Y. Jeremy Dai
# 
# This is an effort to practice my skills learnt from Intro to Machine Learning(Udacity) by building an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.
# 
# ## Intro
# One of largest companies in the US, Enron, collapsed into bankruptcy all of a sudden in 2000. This project is to build a person of interest (POI) identifier based on financial and email data made public as a result of the Enron scandal. A list of POI, who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity, is provided by Udacity for supervised machine learning.
# 
# ## Exploring the data

# In[147]:

import sys
import pickle
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import time
import tester

sys.path.append("../tools/")

#from feature_format import featureFormat, targetFeatureSplit


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Data Exploration
### Check number of samples and POI
print 'Number of Poeple in the sample: ',len(data_dict)

n = 0
for person_name in data_dict:
    if data_dict[person_name]["poi"]==1:
        n+=1
print 'Number of POI: ',n
    
### Create a dataframe based on data_dict, drop the 'email_address' column
enron=pd.DataFrame.from_dict(data_dict, orient='index')
enron=enron.drop('email_address',axis=1) 
print 'There are',enron.shape[1],'features available.'


# In[148]:

### Check the NaN values
enron=enron.replace('NaN',np.nan)
enron_nan=enron.isnull().sum()
enron_nan.sort_values(ascending=False, inplace=True)
print '\nTotal number of NaN values in the dataset:',enron_nan.sum()
print '\nTop five features with most NaN values in the dataset:'
print enron_nan[:5]


# In[149]:

for f in enron_nan[:5].index:
    yes = enron[enron[f].isnull()]['poi'].sum()
    no = enron[~enron[f].isnull()]['poi'].sum()
    print f,'is NaN: there are',yes,'POI'
    print f,'is NOT NaN: there are',no,'POI'
    print 'The ratio of POI in NaN value is', float(yes)/(yes+no)
    print 'The ratio of NaN for',f,'is',float(enron_nan[f])/146,'\n'


# For the top five features with most NaN values in the dataset, we cannot simply remove them since if POI is with NaN value or not for some of these feature does not seem random. We will replace them with 0.

# In[150]:

### use visulization to find outliers
enron.fillna(0,inplace=True)

for i, j in enron[['salary','bonus']].values:
    plt.scatter(i,j)

plt.xlabel('salary')
plt.ylabel('bonus')
plt.show()

print 'The sample with biggest salary is:', list(enron.sort_values(ascending=False, by='salary')[:1].index)


# This is definitely a outlier are we are going to remove it.

# In[151]:

### Remove the clear outlier : 'Total'
try:
    enron=enron.drop('TOTAL') 
except:
    pass


# ## Create new features

# In[152]:

### Reset columns
features_list = ['poi',
        'salary','bonus', 'deferral_payments', 'deferred_income', 'director_fees',
        'expenses', 'loan_advances', 'long_term_incentive', 'other', 'total_payments',
        'restricted_stock', 'restricted_stock_deferred',
        'exercised_stock_options','total_stock_value',
        'to_messages','from_messages','shared_receipt_with_poi',
        'from_poi_to_this_person', 'from_this_person_to_poi',]

enron=enron.reindex_axis(features_list, axis=1)

print 'Create a new feature summing up payments and stocks'
enron['total_finance']=enron['total_payments']+enron['total_stock_value']

print 'Create a new feature of the percentage of person\'s emails to POI to all sent emails'
enron['p_poi_to_msg']=1.0*enron['from_this_person_to_poi']/enron['to_messages']

print 'Create a new feature of the percentage of person\'s emails from POI to all received emails'
enron['p_poi_from_msg']=1.0*enron['from_poi_to_this_person']/enron['from_messages']

enron=enron.fillna(0)

### check the columns
enron.iloc[:5, 20:23]

### update the feature list
features_list = list(enron.columns)


# In[153]:

enron[enron['total_finance']==0]


# There is a row with no entry:'LOCKHART EUGENE E'. It will be removed.

# In[154]:

### Remove 'LOCKHART EUGENE E' Row

try:
    enron=enron.drop('LOCKHART EUGENE E') 
except:
    pass

### Check other outliers
ave=enron['salary'].mean()
poi_h_salary=enron['poi'][enron['salary']>ave]

print '{} of {} people whose salary is larger than the average are POI'.format(sum(poi_h_salary), len(poi_h_salary))


# We cannot simply remove these people. It seems that POI tends to have salary higher than the average. 
# 
# Next, we will select features for training.
# 
# ## Select features

# In[155]:

### Test the created features use Decision Tree

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.cross_validation import train_test_split

features=enron.iloc[:,1:].values
labels=enron.iloc[:,:1].values

### Split the data into training and test
print '70% of data is used for training and the remaining 30% is used for test.'
feature_train, feature_test, label_train, label_test =train_test_split(features, labels,test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier(random_state = 39) 

### Create the function to train and output the evaluation results
def test(clf):
    clf.fit(feature_train, label_train)
    label_pred = clf.predict(feature_test)
    ### Quantitative evaluation of the model quality on the test set
    print 'The evaluation results:'
    print classification_report(label_test, label_pred,target_names=['Not POI','POI'])
    print confusion_matrix(label_test, label_pred)   
    
test(clf)

print '\nThe important features are:'
fea=clf.feature_importances_
n=0
for f in fea:
    if f>0.0:
        print f,features_list[n]
    n+=1


# Since we have an imbalanced sample with much more non-POI than POI, both recall and precision rate are important in evaluation.
# 
# In this study, we will utilize the classification_report and confusion_matrix to generate precision rate, recall rate.
# 
# - Recall = True Positive / (True Positive + False Negative). 
# - Precision=True Positive / (True Positive + False Positive). 
# 
# For the above test:
# Recall:
# - Out of 5 POI, only 1 POI were correctly classified correctly. 
# - Out of 39 Non-POI, 37 of them were 'recalled' from the dataset.
# 
# Precision:
# - Out of 3 ‘selected’ POI, only 1 of them are true POI.
# - Out of 41 ‘selected’ non-POI, only 37 of them are true non-POI.
# 
# 
# Depending on the random_state, sometimes the new created features are the important ones while sometimes they are not. We will try Naive Bayes next.

# In[156]:

### Naive_Bayes
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()

print 'Without new created features'

### Limit the features
features=enron.iloc[:,1:20].values
labels=enron.iloc[:,:1].values

feature_train, feature_test, label_train, label_test =train_test_split(features, labels,test_size=0.3, random_state=42)
    
test(clf)

print '\nWith new created features:'

# Change the features list back with new added features
features=enron.iloc[:,1:].values
labels=enron.iloc[:,:1].values

feature_train, feature_test, label_train, label_test =train_test_split(features, labels,test_size=0.3, random_state=42)

test(clf)


# Based on this test, the new features improved the recall rate for Non-POI. We will include the new features in the following study. 
# 
# We are going to use three machine learning algorithms:
# 
# - Naive Bayes
# - SVC
# - Decision Tree Classifier
# 
# ## Pick and Tune the Algorithms
# 
# An algorithm may be highly sensitive to some of its features. The choose of good parameters may have a dominant effect on the algorithm performance. 
# 
# For example, parameter C plays an important role in SVC algorithm. This parameter is a tradeoff between smooth decision boundary and classifying training points correctly. If it is too low, the algorithm will pay little attention to data and lead to high error on training data. However, if it is too high, it will classify all training examples correctly by providing a really weird boundary. This leads to overfitting and results in high error on testing data.
# 
# In this study, we use GridSearchCV to fine tune the algorithm. I start with default parameters and level it up and down. Based on the GridSearchCV function I will adjust the parameters again. For example, if the GridSearchCV chooses the smallest value for the parameter, I will add a smaller number in the search list.

# In[157]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest


cv = StratifiedShuffleSplit(test_size=0.3, random_state=42)

### Store to my_dataset for easy export below.
my_dataset = enron.to_dict(orient = 'index')

### Define a function using PCA
def find_fit_pca(model, param_grid=[{}]):
    estimators = [
    ('scaler',StandardScaler()),
    ('reduce_dim', PCA(svd_solver='randomized')),
    model]
    pipe = Pipeline(estimators)
    clf = GridSearchCV(pipe, param_grid=param_grid,cv=cv,scoring='f1')
    clf = clf.fit(feature_train, label_train)
    return clf.best_estimator_

### Define a function using SelectKBest
def find_fit_select_k(model, param_grid=[{}]):
    estimators = [
    ('scaler',StandardScaler()),
    ('reduce_dim', SelectKBest()),
    model]
    pipe = Pipeline(estimators)
    clf = GridSearchCV(pipe, param_grid=param_grid,cv=cv,scoring='f1')
    clf = clf.fit(feature_train, label_train)
    return clf.best_estimator_


# Pipelines are used for multi-stage operations.
# 
# StratifiedShuffleSplit is used to for GridSearchCV to find best parameters. Since we have a very imbalanced sample with limited samples, it is easier to have problems like overfitting. To avoid such problem, StratifiedShuffleSplit function will be a good way to rotate training and testing groups and limit the overfitting problem.
# 
# 

# In[158]:

### Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

model=('clf', GaussianNB())

param_grid = dict(reduce_dim__n_components=range(2, 10))

print 'The evaluation results for Gaussian Naive Bayes with PCA:'
clf_nb_pca=find_fit_pca(model,param_grid)
tester.dump_classifier_and_data(clf_nb_pca, my_dataset, features_list)
tester.main()

param_grid = dict(reduce_dim__k=range(3, 10))

print '\nThe evaluation results for Gaussian Naive Bayes with SelectKBest:'
clf_nb_k=find_fit_select_k(model,param_grid)
tester.dump_classifier_and_data(clf_nb_k, my_dataset, features_list)
tester.main()


# In[159]:

### SVC
from sklearn.svm import SVC

model=('clf', SVC())

param_grid = dict(reduce_dim__n_components=[2, 5, 10],
                  clf__C=[1e-20, 1e-10, 0.1],
                  clf__gamma=[1e-30, 1e-15, 0.1])

print 'The evaluation results for SVC with PCA:'
clf_svc_pca=find_fit_pca(model,param_grid)
### User tester function provided by Udacity to evaluate the results
tester.dump_classifier_and_data(clf_svc_pca, my_dataset, features_list)
tester.main()

param_grid = dict(reduce_dim__k=range(2, 6),
                  clf__C=[1e-20, 1e-10, 0.1],
                  clf__gamma=[1e-30, 1e-15, 0.1])

print '\nThe evaluation results for SVC with SelectKBest:'
clf_svc_k=find_fit_select_k(model,param_grid)
### User tester function provided by Udacity to evaluate the results
tester.dump_classifier_and_data(clf_svc_k, my_dataset, features_list)
tester.main()


# In[160]:

### Decision Tree
from sklearn import tree

model=('clf', tree.DecisionTreeClassifier())

param_grid = dict(reduce_dim__n_components=range(2, 10),
                clf__criterion=('gini','entropy'),
              clf__splitter=('best','random'),
              clf__min_samples_split=range(2,10))

print 'The evaluation results for Decision Tree with PCA:'
clf_tree_pca=find_fit_pca(model,param_grid)
tester.dump_classifier_and_data(clf_tree_pca, my_dataset, features_list)
tester.main()

param_grid = dict(reduce_dim__k=range(2, 10),
                clf__criterion=('gini','entropy'),
              clf__splitter=('best','random'),
              clf__min_samples_split=range(2,10))

print '\nThe evaluation results for Decision Tree with SelectKBest:'
clf_tree_k=find_fit_select_k(model,param_grid)
tester.dump_classifier_and_data(clf_tree_k, my_dataset, features_list)
tester.main()


# Take the process of fine-tuning Decision Tree algorithm as an example. Two criterion -- 'gini' and 'entropy' are tried. Two types of splitter -- 'best' and 'random' are compared. And the min_samples_split can be 2,3,4,5,6,7,8,9 and 10 . Using SelectKBest,  GridSearchCV gives us the best DecisionTreeClassifier as
# (criterion='entropy', min_samples_split=4, splitter='random'))
# 
# 
# ## Conclusion
# Before feature selection, I use StandardScaler for feature normalization. Scaling would help the feature selection since the ranges of available features vary a lot.
#  
# After scaling, I tried two methods to trim down the number of features. One is Principal Component Analysis(PCA), and the other one is an automated feature selection function SelectKBest.
#  
# Based on my tests, PCA does not work well combined with the models I tried. It may be because we set all the missing data to 0 and get the variations of each feature muddied.

# In[161]:

print 'After running tester.py, the performance measurements are summarized in the following table:'
pd.DataFrame([[0.84393, 0.38069, 0.27200, 0.31729],
              [0.85767, 0.45386, 0.33200, 0.38348],
              [0.80533, 0.24586, 0.22250, 0.23360],
              [0.84347, 0.38642, 0.29600, 0.33522]],
             columns = ['Accuracy','Precision', 'Recall', 'F1'], 
             index = ['Gaussian Naive Bayes with PCA','Gaussian Naive Bayes with SelectKBest','Decision Tree with PCA', 'Decision Tree with SelectKBest'])


# Validation is the process that tests if the model works well. One classic mistake is over-fitting. It means the model has a high variance and pays too much attention to data. Since it is very perceptive to data，it does not work well on the test data and produces much higher error on test data than on training data.
# 
# ### The best performer is Gaussian Naive Bayes Classifer.
# The features are selected using SelectKest. 

# In[162]:

### Get the features and scores
f_list=list(clf_nb_k.named_steps['reduce_dim'].get_support())
f_score=list(clf_nb_k.named_steps['reduce_dim'].scores_)

fea_list = list(enron.columns[1:])

### print the bar graph
print 'Bar plot of the feature scores:'
f_s=zip(fea_list,f_score)
f_s=sorted(f_s, key=lambda x:x[1], reverse=True)
f,s = zip(*f_s)
plt.bar(range(len(f)),s,tick_label=f)
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Score')
plt.show()


# The score for bonus is the most important feature among the all, followed by salary, total finance and total stock value. It makes sense since the aim of POI in the enron fraud is to gain their finanical benefits.

# In[163]:

for i, fea,s in zip(f_list,fea_list,f_score):
    if i:
        print fea, s


# In[164]:

### Dump the classifier, dataset, and features_list for review by Udacity
tester.dump_classifier_and_data(clf_nb_k, my_dataset, features_list)


# In[165]:

features_list

