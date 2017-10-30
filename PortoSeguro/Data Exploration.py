

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Train-data-load" data-toc-modified-id="Train-data-load-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Train data load</a></div><div class="lev2 toc-item"><a href="#Data-preparation" data-toc-modified-id="Data-preparation-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Data preparation</a></div><div class="lev3 toc-item"><a href="#Imputting-missing-values" data-toc-modified-id="Imputting-missing-values-111"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Imputting missing values</a></div><div class="lev1 toc-item"><a href="#Model-selection" data-toc-modified-id="Model-selection-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Model selection</a></div><div class="lev2 toc-item"><a href="#Feature-selection" data-toc-modified-id="Feature-selection-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Feature selection</a></div><div class="lev2 toc-item"><a href="#Evaluating--single-classifiers" data-toc-modified-id="Evaluating--single-classifiers-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Evaluating  single classifiers</a></div><div class="lev3 toc-item"><a href="#Random-Forest" data-toc-modified-id="Random-Forest-221"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Random Forest</a></div><div class="lev3 toc-item"><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-222"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Logistic Regression</a></div><div class="lev3 toc-item"><a href="#XGBoost" data-toc-modified-id="XGBoost-223"><span class="toc-item-num">2.2.3&nbsp;&nbsp;</span>XGBoost</a></div><div class="lev1 toc-item"><a href="#Predictions-on-test-set" data-toc-modified-id="Predictions-on-test-set-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Predictions on test set</a></div><div class="lev2 toc-item"><a href="#Missing-values-imputation-and-feature-engineering" data-toc-modified-id="Missing-values-imputation-and-feature-engineering-31"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Missing values imputation and feature engineering</a></div><div class="lev2 toc-item"><a href="#Predict-on-test-and-output-submission" data-toc-modified-id="Predict-on-test-and-output-submission-32"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Predict on test and output submission</a></div>


import os
from datetime import datetime

import pandas as pd

from scipy.stats import ttest_ind 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, f1_score, make_scorer, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import xgboost
from xgboost import XGBClassifier

from imblearn.over_sampling import RandomOverSampler



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# # Train data load


d = pd.read_csv('datasets' + os.sep + 'train.csv', na_values=-1)



d.shape



n_rows = d.shape[0];



d.head()


# Drop _id_ column


del d['id']



d.columns.values


# How much missing data do we have?


d.isnull().shape


# All examples have at least one missing feature.
# What are the missing features?


columns_with_na = d.columns[d.isnull().any()].tolist()
columns_with_na


# Let's create dummies for all categorical variables, that __do not__ have missing values (we have to yet impute the missing values). We create dummies now, because first we will need then in main model anyway and second we will need then in models predicting missing values in other predictor columns.   


categorical_columns = [col for col in d.columns if col.endswith('_cat') and col not in columns_with_na]
d = pd.get_dummies(d, columns=categorical_columns, drop_first=True);



d.columns.values


# How much classes are in balance?


d.groupby('target').size() / d.shape[0]


# Claims are filled for __3.64%__ policies.

# __TODO__: try oversampling, later.

# ## Data preparation
# ### Imputting missing values
# Let's go thru all features with missing values one by one, and determine simple imputation strategy that makes most sense at first glance.
# 
# For every feature with missing data we will look into:
# 1. Distribution of non-missing data of that features
# 2. Is data _missingness_ related to filling the claim more often? (We suspect Missing At Random - MAR).
# 3. Is it possible to build a model (regression for numerical and classification for categorical) to train it on filled examples and use to predict missing values?
# 4. Or determine what other imputation strategy can we use. 

# Some utilities that we will use while looking at data:


columns_with_na_categorical = [c for c in columns_with_na if c.endswith('_cat')]
columns_with_na_numeric = [c for c in columns_with_na if not c.endswith('_cat')]



def print_stats(col):
    missing_cnt = d[d[col].isnull()].shape[0]
    print('Missing values: {:.2f}%'.format(100 * missing_cnt / n_rows))
    print('Missing count: {}'.format(missing_cnt))
    if col.endswith('_cat'):
        print(d.groupby(col).size())
    else:
        print(d[col].describe())



def ttest(col):
    ttest = ttest_ind(d[d[col].isnull()].target, d[d[col].notnull()].target, equal_var=False)
    print(col + ': ' + str(ttest))



def fillWithMean(col, df=d):
    df.fillna({col: df[col].mean()}, inplace=True);
    
def fillWithMode(col, df=d):
    df.fillna({col: df[col].mode()[0]}, inplace=True); # mode() returns a one element series


# Regression model to fill missing oridinal and continuous data


def predict_missing(estimator, dependent_col, df=d):
    columns_with_na = df.columns[df.isnull().any()].tolist()
    predictor_cols = [c for c in df.columns if c not in columns_with_na and c != dependent_col and c != 'target']
    d_rows_na = df[df[dependent_col].isnull()]
    d_rows_no_na = df[df[dependent_col].notnull()]
    estimator.fit(d_rows_no_na[predictor_cols], d_rows_no_na[dependent_col])
    predicted = estimator.predict(d_rows_na[predictor_cols])
    cv_scores = cross_validate(estimator, d_rows_no_na[predictor_cols], d_rows_no_na[dependent_col], n_jobs=-1, return_train_score=False)['test_score']
    return predicted, cv_scores


# Let's go thru all features one by one:


columns_with_na_numeric


# __ps_reg_03__


print_stats('ps_reg_03')



predicted, errs = predict_missing(estimator=LinearRegression(), dependent_col = 'ps_reg_03')
errs



ttest('ps_reg_03')


# There is correlation b/w data missingness and positive class. What is the distribution of data for positive class for non-missing values?


d[d['target']==1]['ps_reg_03'].describe()


# Values are higher. But not much. Use regression model for now, for lack of better idea:


d.loc[d['ps_reg_03'].isnull(), 'ps_reg_03'] = predicted


# __ps_car_11__


print_stats('ps_car_11')


# It is only 5 missing values. Does not matter much anyway.


predicted, errs = predict_missing(estimator=LinearRegression(), dependent_col = 'ps_car_11')
errs



ttest('ps_car_11')


# Regression is not accurate, but let's fill from regression:


d.loc[d['ps_car_11'].isnull(), 'ps_car_11'] = predicted


# __ps_car_12__


print_stats('ps_car_12')


# Only 1 missing value.


predicted, errs = predict_missing(estimator=LinearRegression(), dependent_col = 'ps_car_12')
errs


# Quite good regression accuracy.


d.loc[d['ps_car_12'].isnull(), 'ps_car_12'] = predicted


# __ps_reg_14__


print_stats('ps_car_14')



predicted, errs = predict_missing(estimator=LinearRegression(), dependent_col = 'ps_car_14')
errs


# Very poor regression accuracy, fill with mean:


fillWithMean('ps_car_14')


# And now let's go thru all categorical  features missing


columns_with_na_categorical


# __ps_ind_02_cat__


print_stats('ps_ind_02_cat')



# predicted, errs = predict_missing(DecisionTreeClassifier(), 'ps_ind_02_cat')
# print(predicted, errs)


# What is the distribution for positive target class?


d[d['target']==1].groupby('ps_ind_02_cat').size()


# Classifier accuracy not very good,  filling with mode for a moment. Also the number of samples with that feature missing is not very big.


fillWithMode('ps_ind_02_cat')


# __ps_ind_04_cat__


print_stats('ps_ind_04_cat')



d[d['target']==1].groupby('ps_ind_04_cat').size()



# predicted, errs = predict_missing(DecisionTreeClassifier(), 'ps_ind_04_cat')
# print(predicted, errs)


# Classifier accuracy was poor, filling with mode


fillWithMode('ps_ind_04_cat')


# __ps_ind_05_cat__


print_stats('ps_ind_05_cat')



predicted, errs = predict_missing(DecisionTreeClassifier(), 'ps_ind_05_cat')
print(predicted, errs)



d.loc[d['ps_ind_05_cat'].isnull(), 'ps_ind_05_cat'] = predicted


# __ps_car_01_cat__


print_stats('ps_car_01_cat')



# predicted, errs = predict_missing(DecisionTreeClassifier(), 'ps_car_01_cat')
# print(predicted, errs)


# Very poor accuracy, fill with mode.


fillWithMode('ps_car_01_cat')


# __ps_car_02_cat__


print_stats('ps_car_02_cat')



predicted, errs = predict_missing(DecisionTreeClassifier(), 'ps_car_02_cat')
print(predicted, errs)



d.loc[d['ps_car_02_cat'].isnull(), 'ps_car_02_cat'] = predicted


# __ps_car_03_cat__


print_stats('ps_car_03_cat')


# Most data is missing here. What is data distribution in positive target class?


d[d['target']==1].groupby('ps_car_03_cat').size()


# Seems very influencing..
# Are we able to build any reliable classifier from other features?


predicted, errs = predict_missing(DecisionTreeClassifier(), 'ps_car_03_cat')
print(predicted, errs)



d.loc[d['ps_car_03_cat'].isnull(), 'ps_car_03_cat'] = predicted


# __ps_car_05_cat__


print_stats('ps_car_05_cat')



d[d['target']==1].groupby('ps_car_05_cat').size()



# predicted, errs = predict_missing(DecisionTreeClassifier(), 'ps_car_05_cat')
# print(predicted, errs)


# Poor classifier accuracy.
# 
# __Do not use that feature__. Too many items missing, and does not seem to have any correlation to target class.


del d['ps_car_05_cat']


# __ps_car_07_cat__


print_stats('ps_car_07_cat')



predicted, errs = predict_missing(DecisionTreeClassifier(), 'ps_car_07_cat')
print(predicted, errs)



d.loc[d['ps_car_07_cat'].isnull(), 'ps_car_07_cat'] = predicted


# __ps_car_09_cat__


print_stats('ps_car_09_cat')



# predicted, errs = predict_missing(DecisionTreeClassifier(), 'ps_car_09_cat')
# errs


# Poor classifier accuracy, fill with mode


fillWithMode('ps_car_09_cat')


# Have we handled all missing data?


columns_with_na = d.columns[d.isnull().any()].tolist()
assert not columns_with_na


# Now we can create dumies also for the categorical variables that had missing values.


d = pd.get_dummies(d, columns=[c for c in columns_with_na_categorical if c != 'ps_car_05_cat'], drop_first=True);


# # Model selection

# ## Feature selection
# What are features correlations?


binary_features = [c for c in d.columns if '_bin' in c]
binary_ind_features = [c for c in binary_features if '_ind_' in c]
binary_calc_features = [c for c in binary_features if '_calc_' in c]
categorical_features = [c for c in d.columns if '_cat' in c and '_calc_' not in c]
numeric_features = [c for c in d.columns if c not in binary_features and c not in categorical_features and '_calc_' not in c and c!='target']


# Draw correlation matrix


corr = d[binary_features + numeric_features + ['target']].corr()
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, ax=ax, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()


# We can see strong correlations for couple of features, let's zoom into them


high_correlations = d[['ps_ind_14', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_14']].corr()
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(high_correlations, ax=ax, 
            xticklabels=high_correlations.columns.values,
            yticklabels=high_correlations.columns.values)
plt.show()


# We will have to somehow merge that features (PCA?)

# What are the correlations of binary and numerical columns to the target variable?


corr['target']


# We clearly see that _calc_ featrues do not correlate at all - they will not be taken as features. Some other features have also minimal correlation. They will be eliminated by PCA method later anyway.

# What features are the most important by PCA method?


pca = PCA()
features = [c for c in d.columns if c != 'target']
pca.fit(d[features]);



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
ax1.plot(pca.explained_variance_ratio_)
ax2.plot(pca.explained_variance_ratio_[:25])
plt.show()



pca.components_


# ## Evaluating  single classifiers

# The final model performance metric will be Gini index. 
# 
# However we cannot pass that function to cross_validate / grid_search routines, as that routines run in parallel and it is not possible to pickle a function (without much complication). Instead our target metric will be just Area Under ROC, as gini index is directly proportional to it. Only at the very end we will compute gini normalized, in one thread. 


def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) -1
    return g

def gini_norm(y, pred):
    normed = gini(y, pred) / gini(y,y)
    return normed



features = binary_ind_features + categorical_features + numeric_features
X = d[features]
y = d['target']



# def gini(actual, pred):
#     assert (len(actual) == len(pred))
#     all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
#     all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
#     totalLosses = all[:, 0].sum()
#     giniSum = all[:, 0].cumsum().sum() / totalLosses

#     giniSum -= (len(actual) + 1) / 2.
#     return giniSum / len(actual)


# def gini_normalized(actual, pred):
#     return gini(actual, pred) / gini(actual, actual)


# ### Random Forest


# classifier = Pipeline([
#     ('pca', PCA()),
#     ('tree', RandomForestClassifier(n_estimators=50, class_weight = 'balanced'))
# ])

# grid_search_CV = GridSearchCV(classifier, {
#     'pca__n_components': [25, 50, 100],
#     'tree__min_samples_split': [1000, 5000, 10000],
# }, n_jobs=7, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring=make_scorer(roc_auc_score), verbose=10)

# grid_search_CV.fit(X, y)

# grid_search_CV.best_params_, grid_search_CV.best_score_

# # {'pca__n_components': 25,
# #  'tree__class_weight': 'balanced',
# #  'tree__min_samples_split': 10}



# classifier = Pipeline([
#     ('tree', RandomForestClassifier(n_estimators=50, class_weight = 'balanced', criterion='entropy', min_samples_split=5000))
# ])

# grid_search_CV = GridSearchCV(classifier, {
#     'tree__min_samples_split': [3000, 5000, 7000, 9000],
#     'tree__max_features': [13, 14, 15]
# }, n_jobs=7, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring=make_scorer(roc_auc_score), verbose=10)

# grid_search_CV.fit(X, y)

# grid_search_CV.best_params_


# ### Logistic Regression


# from sklearn.linear_model import LogisticRegression
# import numpy as np

# grid_search_CV = GridSearchCV(LogisticRegression(class_weight='balanced'), {
#     'C': np.arange(0.001, 0.01, 0.001)
# }, n_jobs=-1, cv=StratifiedKFold(n_splits=4, shuffle=True), scoring=make_scorer(roc_auc_score), verbose=10)

# grid_search_CV.fit(X, y)

# grid_search_CV.best_params_, grid_search_CV.best_score_



# cross_validate(LogisticRegression(class_weight = 'balanced', C=0.002), X, y, n_jobs=1, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring=make_scorer(roc_auc_score))
# cross_validate(LogisticRegression(class_weight = 'balanced', C=0.002), X, y, n_jobs=1, cv=StratifiedKFold(n_splits=4, shuffle=True), scoring=make_scorer(gini_norm))
# # roc_auc_score round 0.59, corresponsing to gini of 0.18



# trees = RandomForestClassifier(n_estimators=100, class_weight = 'balanced', criterion='entropy', min_samples_split=5000, n_jobs=-1)
# cross_validate(trees, X, y, n_jobs=1, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring=make_scorer(gini_norm))
# gini of 0.18


# ### XGBoost


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=3, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgboost.DMatrix(dtrain[predictors].values, label=dtrain['target'].values)
        cvresult = xgboost.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['target'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("Accuracy : %.4g" % accuracy_score(dtrain['target'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % roc_auc_score(dtrain['target'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')



# xgb1 = XGBClassifier(learning_rate=0.05, n_estimators=1000, max_depth=3, min_child_weight=2, gamma=0.01, reg_alpha=10,
#                                             subsample=0.9, colsample_bytree=0.9, objective='binary:logistic')

# modelfit(xgb1, d, features)



# grid_search_CV = GridSearchCV(XGBClassifier(learning_rate=0.2, n_estimators=150, max_depth=3, min_child_weight=2, gamma=0.01, reg_alpha=10,
#                                             subsample=0.9, colsample_bytree=0.9, objective='binary:logistic'), {
#     'reg_alpha':[20, 50, 150]
# }, n_jobs=-1, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring='roc_auc', verbose=10)

# grid_search_CV.fit(X, y)

# grid_search_CV.best_params_, grid_search_CV.best_score_



# classifier = XGBClassifier(learning_rate=0.05, n_estimators=750, max_depth=3, min_child_weight=2, gamma=0.01, reg_alpha=10,
#                                             subsample=0.9, colsample_bytree=0.9, objective='binary:logistic')
# cross_validate(classifier, X, y, n_jobs=1, cv=StratifiedKFold(n_splits=4, shuffle=True), scoring=make_scorer(gini_norm))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
classifier = XGBClassifier(learning_rate=0.05, n_estimators=750, max_depth=3, min_child_weight=2, gamma=0.01, reg_alpha=10,
                                            subsample=0.9, colsample_bytree=0.9, objective='binary:logistic')
classifier.fit(X_train, y_train)

y_pred = classifier.predict_proba(X_test)[:,1]



gini_norm(y_test, y_pred)
# 0.2837


# ### Minority class oversampling


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# X_resampled, y_resampled = RandomOverSampler().fit_sample(X_train, y_train)



# X_resampled_df = pd.DataFrame(X_resampled, columns = X_train.columns)
# y_resampled_df = pd.Series(y_resampled)



# # WARN: this tuned settings may not be optimal for oversampling
# classifier = XGBClassifier(learning_rate=0.05, n_estimators=750, max_depth=3, min_child_weight=2, gamma=0.01, reg_alpha=10,
#                                             subsample=0.9, colsample_bytree=0.9, objective='binary:logistic')
# classifier.fit(X_resampled_df, y_resampled_df)

# y_pred = classifier.predict_proba(X_test)[:,1]
# gini_norm(y_test, y_pred)


# The oversampled score of 0.2764 is less than baseline 0.2837. Plain oversampling did not help for XGB, maybe it will help for RF. 

# Try out RF with random oversampling


# classifier = RandomForestClassifier(n_estimators=100, class_weight = 'balanced', criterion='entropy', min_samples_split=5000, n_jobs=-1)
# classifier.fit(X_resampled_df, y_resampled_df)

# y_pred = classifier.predict_proba(X_test)[:,1]
# gini_norm(y_test, y_pred)


# Far better than RF on unbalanced dataset (which had Gini of 0.18). To be used in further ensembles.

# Try out SMOTE


# from imblearn.over_sampling import SMOTE
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train)



# X_resampled_df = pd.DataFrame(X_resampled, columns = X_train.columns)
# y_resampled_df = pd.Series(y_resampled)



# classifier = XGBClassifier(learning_rate=0.05, n_estimators=750, max_depth=3, min_child_weight=2, gamma=0.01, reg_alpha=10,
#                                             subsample=0.9, colsample_bytree=0.9, objective='binary:logistic')
# classifier.fit(X_resampled_df, y_resampled_df)

# y_pred = classifier.predict_proba(X_test)[:,1]
# gini_norm(y_test, y_pred)


# Very poor: 0.1982

# SMOTE on RF?


# classifier = RandomForestClassifier(n_estimators=100, class_weight = 'balanced', criterion='entropy', min_samples_split=5000, n_jobs=-1)
# classifier.fit(X_resampled_df, y_resampled_df)

# y_pred = classifier.predict_proba(X_test)[:,1]
# gini_norm(y_test, y_pred)


# Poor: 0.1891

# Under sampling?


# from imblearn.under_sampling import RandomUnderSampler
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# X_resampled, y_resampled = RandomUnderSampler().fit_sample(X_train, y_train)
# X_resampled_df = pd.DataFrame(X_resampled, columns = X_train.columns)
# y_resampled_df = pd.Series(y_resampled)

# classifier = RandomForestClassifier(n_estimators=100, class_weight = 'balanced', criterion='entropy', min_samples_split=5000, n_jobs=-1)
# classifier.fit(X_resampled_df, y_resampled_df)

# y_pred = classifier.predict_proba(X_test)[:,1]
# gini_norm(y_test, y_pred)


# Not bad 0.2315 vs 0.19 baseline


# from imblearn.over_sampling import ADASYN
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# X_resampled, y_resampled = ADASYN().fit_sample(X_train, y_train)
# X_resampled_df = pd.DataFrame(X_resampled, columns = X_train.columns)
# y_resampled_df = pd.Series(y_resampled)

# classifier = RandomForestClassifier(n_estimators=100, class_weight = 'balanced', criterion='entropy', min_samples_split=5000, n_jobs=-1)
# classifier.fit(X_resampled_df, y_resampled_df)

# y_pred = classifier.predict_proba(X_test)[:,1]
# gini_norm(y_test, y_pred)


# Poor ADASYN with RF: 0.1684

# ### Extra trees


# classifier = ExtraTreesClassifier(n_estimators=24, class_weight = 'balanced', criterion='entropy', min_samples_split=5000, n_jobs=-1, max_features=14)

# grid_search_CV = GridSearchCV(classifier, {
#     'min_samples_split': [100, 1000, 5000, 9000],
#     'max_features': [12, 14, 16]
# }, n_jobs=7, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring=make_scorer(roc_auc_score), verbose=10)

# grid_search_CV.fit(X, y)

# grid_search_CV.best_params_, grid_search_CV.best_score_
# ({'max_features': 14, 'min_samples_split': 5000}, 0.59043612417733271)



# from sklearn.ensemble import ExtraTreesClassifier

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# classifier = ExtraTreesClassifier(n_estimators=100, class_weight = 'balanced', criterion='entropy', min_samples_split=5000, n_jobs=-1, max_features=14)
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict_proba(X_test)[:,1]
# gini_norm(y_test, y_pred)
# # 0.2557


# Extra trees Gini 0.2557


# classifier = ExtraTreesClassifier(n_estimators=100, class_weight = 'balanced', criterion='entropy', min_samples_split=5000, n_jobs=-1, max_features=14)
# classifier.fit(X_resampled_df, y_resampled_df)

# y_pred = classifier.predict_proba(X_test)[:,1]
# gini_norm(y_test, y_pred)


# No gain: 0.2556

# ### Stacking ensembles


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# xgb = XGBClassifier(learning_rate=0.05, n_estimators=750, max_depth=3, min_child_weight=2, gamma=0.01, reg_alpha=10,
#                                             subsample=0.9, colsample_bytree=0.9, objective='binary:logistic')
# xgb.fit(X_train, y_train)

# X_resampled, y_resampled = RandomOverSampler().fit_sample(X_train, y_train)
# X_resampled_df = pd.DataFrame(X_resampled, columns = X_train.columns)
# y_resampled_df = pd.Series(y_resampled)
# rf = RandomForestClassifier(n_estimators=100, class_weight = 'balanced', criterion='entropy', min_samples_split=5000, n_jobs=-1)
# rf.fit(X_resampled_df, y_resampled_df)

# xtrees = ExtraTreesClassifier(n_estimators=100, class_weight = 'balanced', criterion='entropy', min_samples_split=5000, n_jobs=-1, max_features=14)
# xtrees.fit(X_train, y_train)



# xgb_resampled = XGBClassifier(learning_rate=0.05, n_estimators=750, max_depth=3, min_child_weight=2, gamma=0.01, reg_alpha=10,
#                                             subsample=0.9, colsample_bytree=0.9, objective='binary:logistic')
# xgb_resampled.fit(X_resampled_df, y_resampled_df)



# y_pred_xgb = xgb.predict_proba(X_test)[:,1]
# y_pred_rf = rf.predict_proba(X_test)[:,1]
# y_pred_xtrees = xtrees.predict_proba(X_test)[:,1]
# y_pred_xgb_resampled = xgb_resampled.predict_proba(X_test)[:,1]



# import numpy as np

# best_single_classifer_score = 0.2837
# weights_xgb = np.arange(0.95, 1, 0.005)
# max_improvement_over_best = 0
# w_xgb_argmax = 0
# w_xgb_resampled_argmax = 0
# w_rf_argmax = 0
# w_xtrees_argmax = 0
# for weight_xgb in weights_xgb:
#     parts = np.arange(0.1, 1, 0.05)
#     for weight_rf in [(1 - weight_xgb) * part for part in parts ]:
#         parts2 = np.arange(0.1, 1, 0.05)
#         for weight_xgb_resampled in [(1 - (weight_xgb + weight_rf)) * part for part in parts2 ]:
#             weight_xtrees = 1 - (weight_xgb + weight_rf + weight_xgb_resampled)
#             averaged_preds = weight_xgb * y_pred_xgb + weight_rf * y_pred_rf + weight_xgb_resampled * y_pred_xgb_resampled + weight_xtrees * y_pred_xtrees
#             g = gini_norm(y_test, averaged_preds)
#             better_by = g - best_single_classifer_score
#             if (better_by > max_improvement_over_best):
#                 max_improvement_over_best = better_by
#                 w_xgb_argmax = weight_xgb
#                 w_xgb_resampled_argmax = weight_xgb_resampled
#                 w_rf_argmax = weight_rf
#                 w_xtrees_argmax = weight_xtrees
#                 print('Best improvement: ' + str(max_improvement_over_best), w_xgb_argmax, w_xgb_resampled_argmax, w_rf_argmax, w_xtrees_argmax)


# # Predictions on test set


d_test = pd.read_csv('datasets' + os.sep + 'test.csv', na_values=-1)



d_test.shape


# ## Missing values imputation and feature engineering


submission = pd.DataFrame()
submission['id'] = d_test['id']
del d_test['id']



d_test = pd.get_dummies(d_test, columns=categorical_columns, drop_first=True);



def fillFromModel(colname, df, estimator):
    if df[colname].isnull().values.any():
        predicted, _ = predict_missing(estimator, dependent_col = colname, df=df)
        df.loc[d_test[colname].isnull(), colname] = predicted



fillFromModel('ps_reg_03', d_test, LinearRegression())
fillFromModel('ps_car_11', d_test, LinearRegression())
fillFromModel('ps_car_12', d_test, LinearRegression())

fillWithMean('ps_car_14', d_test)
fillWithMode('ps_ind_02_cat', d_test)
fillWithMode('ps_ind_04_cat', d_test)

fillFromModel('ps_ind_05_cat', d_test, DecisionTreeClassifier())

fillWithMode('ps_car_01_cat', d_test)

fillFromModel('ps_car_02_cat', d_test, DecisionTreeClassifier())
fillFromModel('ps_car_03_cat', d_test, DecisionTreeClassifier())

del d_test['ps_car_05_cat']

fillFromModel('ps_car_07_cat', d_test, DecisionTreeClassifier())

fillWithMode('ps_car_09_cat', d_test)

d_test = pd.get_dummies(
    d_test,
    columns=[c for c in columns_with_na_categorical if c != 'ps_car_05_cat'],
    drop_first=True)


# Any other NAs in test?


test_set_columns_with_na = d_test.columns[d_test.isnull().any()].tolist()
assert not test_set_columns_with_na


# ## Predict on test and output submission


submission['target'] = classifier.predict_proba(d_test[features])[:,1]



date_time_stamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
submission.to_csv('submissions' + os.sep + 'submission_' + date_time_stamp + '.csv', float_format='%.4f', index=False)

