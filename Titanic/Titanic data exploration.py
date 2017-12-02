

# TODO
# - are name lengths not misleading in many cases - or it is a feature actually - try throwing out pieces in parens.
# - try less features! And check on other test set
# - feature: correlate ticket numbers
# - cabins: decks, numer of cabin
# - better plotting with Seaborn: like in published kernels - bars, not distribution plot


import os
from datetime import datetime

import pandas as pd
import numpy as np

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



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


# ## Train data load


d = pd.read_csv('datasets' + os.sep + 'train.csv', na_values=-1)



d.shape



d.head(15)



del d['PassengerId']



d = pd.get_dummies(d, columns=['Sex', 'Embarked']);


# ## Data exploration

# #### Fill missing values


d.columns[d.isnull().any()]


# - age


d[d['Age'].isnull()].shape



d[d['Age'].isnull()]


# How average age differs by pclass and sex?


d.groupby(['Pclass', 'Sex_female']).agg({'Age': 'mean'})


# Differs much by pclass and sex, use it

# What is the relation between title (Mr. Mrs. Miss, etc) and age?


d['isMr'] = d['Name'].str.contains('Mr\.')
d['isMrs'] = d['Name'].str.contains('Mrs\.') # married woman
d['isMaster'] = d['Name'].str.contains('Master\.')
d['isMiss'] = d['Name'].str.contains('Miss\.') | d['Name'].str.contains('Mlle\.')
d['isMs'] = d['Name'].str.contains('Ms\.') # unmarried woman

d['isRev'] = d['Name'].str.contains('Rev\.') # clergy
d['isDr'] = d['Name'].str.contains('Dr\.') # rather as academic grade
d['isNoble'] = d['Name'].str.contains('Lady\.') | d['Name'].str.contains('Mme\.') | d['Name'].str.contains('Sir\.') | d['Name'].str.contains('Countess\.') | d['Name'].str.contains('Don\.') | d['Name'].str.contains('Dona\.') | d['Name'].str.contains('Jonkheer\.')
d['isMilitary'] = d['Name'].str.contains('Major\.') | d['Name'].str.contains('Col\.') | d['Name'].str.contains('Capt\.')

assert d[~d['isMr'] & ~d['isMrs'] & ~d['isMaster'] & ~d['isMiss'] & ~d['isMs'] & ~d['isRev'] & ~d['isDr'] & ~d['isNoble'] & ~d['isMilitary']].empty



honoric_features = ['isMr', 'isMrs', 'isMaster', 'isMiss', 'isMs', 'isRev', 'isDr', 'isNoble', 'isMilitary']



d[honoric_features +['Survived']].corr()['Survived']


# It was checked, by CV that adding honoric features do not help CV.


for f in honoric_features:
    print(f, d[d[f]==True]['Age'].mean(), d[d[f]==True].shape[0])



sns.set(style="ticks")
sns.pairplot(d[d['Age'].notnull()], size=6, hue='Survived', vars=['Age'])
plt.show()



for pclass in [1,2,3]:
    for title in ['isMr', 'isMrs', 'isMaster', 'isMiss']:
        d.loc[(d[title]==True) & (d['Pclass']==pclass) & (d['Age'].isnull()), ['Age']] = d[(d[title]==True) & (d['Pclass']==pclass)]['Age'].mean()
# default
d['Age'].fillna(d['Age'].mean(), inplace=True)



d[['Age', 'Survived']].corr()


# - Cabin


d['Cabin'].head(15)



d['Cabin_filled'] = d['Cabin'].notnull()



d[['Cabin_filled', 'Survived']].corr()


# Target variable distribution


d['Survived'].value_counts()



d.columns


# ### Correlations


features = ['Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Cabin_filled']



sns.set(style="ticks")
sns.pairplot(d, hue='Survived', vars=features)
plt.show()


# Correlation strength


corr = d[features + ['Survived']].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, ax=ax, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()
corr



X = d[features]
y = d['Survived']



cross_validate(RandomForestClassifier(class_weight = 'balanced'), X, y, cv=StratifiedKFold(n_splits=4, shuffle=True), n_jobs=-1)


# ### Classes separability
# 
# How separable are target classes, when looked at after reducing dimensionality (manifolds)?


# # commented, as takes time to execute:
# from sklearn import preprocessing
# from sklearn.manifold import TSNE

# green = y==1
# blue = y==0

# X_scaled = preprocessing.scale(X)

# manifold = TSNE().fit_transform(X_scaled)

# plt.figure(figsize=(12,10))
# plt.scatter(manifold[green,0], manifold[green,1], c='g')
# plt.scatter(manifold[blue,0], manifold[blue,1], c='b')
# plt.show()


# tSNE representation:
# 
# !['tSNE '](img/tSNE.png)

# ## Feature engineering

# Will we get better correlation from dummies from 'Pclass', instead of when treating it as oridinal variable? 


d['Pclass_1'] = d['Pclass']==1
d['Pclass_2'] = d['Pclass']==2
d['Pclass_3'] = d['Pclass']==3



d[['Pclass', 'Pclass_1', 'Pclass_2', 'Pclass_3'] + ['Survived']].corr()



features_tested = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
cross_validate(RandomForestClassifier(class_weight = 'balanced'), d[features_tested], y, cv=StratifiedKFold(n_splits=4, shuffle=True), n_jobs=-1)


# Not better, also as relation is linear.

# - family size


d['familySize'] = d['SibSp'] + d['Parch'] + 1



sns.set(style="ticks")
sns.pairplot(d, size=6, hue='Survived', vars=['familySize'])
plt.show()


# We see that:
# - the greatest change to survive is mid-size family
# - both travelling alone and bigger families have less chance to survive.
# 
# Let's create 3 features for family size - third dummy is actually not needed.


d['family_isAlone'] = d['familySize'] == 1 
d['family_big'] = d['familySize'] > 4 


# - babies?


# d['isBaby']=d['Age']<1
# CV scores not better, also already included in Age feature


# - log transform skewed distributions


plt.hist(d['Fare'])
plt.show()



d['Fare_log'] = np.log2(d['Fare'] + 1)



plt.hist(d['Fare_log'])
plt.show()


# Checked by CV score and accuracy with Fare log transformed is better by good 1% for RF. But not for logistic!


# d[(d['Age']<16) & (d['Parch']==0)& (d['Survived']==1)].shape[0] / d[(d['Age']<16) & (d['Parch']==0)].shape[0]
# d['isSingleChild'] = d.apply(lambda r: r['Age']<16 and r['Parch']==0, axis=1)
# did not help in CV


# - lenght of name


d['isMr_NameLength'] = d.apply(lambda r: len(r['Name'].split(' '))-1 if r['isMr']==True else 0, axis=1)
d['isMrs_NameLength'] = d.apply(lambda r: len(r['Name'].split(' '))-1 if r['isMrs']==True else 0, axis=1)
d['isMaster_NameLength'] = d.apply(lambda r: len(r['Name'].split(' '))-1 if r['isMaster']==True else 0, axis=1)
d['isMiss_NameLength'] = d.apply(lambda r: len(r['Name'].split(' '))-1 if r['isMiss']==True else 0, axis=1)


# __They help Logit by 1% point stable.__


# d[(d['isMiss']) & (d['Pclass']==3)][['isMiss_NameLength', 'Survived']].corr()



d['isMr_LongName_Pclass_1'] = d.apply(lambda r: len(r['Name'].split(' '))-1 >=4 if r['isMr']==True and r['Pclass']==1 else False, axis=1)
d['isMaster_LongName_Pclass_3'] = d.apply(lambda r: len(r['Name'].split(' '))-1 >=4 if r['isMaster']==True and r['Pclass']==3 else False, axis=1)


# - decks


# d['Cabin_first_letter'] = d[d['Cabin'].notnull()]['Cabin'].str[:1]
# d = pd.get_dummies(d, columns=['Cabin_first_letter']);



# cabin_code_features = ['Cabin_first_letter_' + l for l  in ['A', 'B', 'C', 'D', 'E', 'F']]
# d[cabin_code_features + ['Survived']].corr()



# d[d['Pclass']==3][['Cabin_first_letter_F', 'Survived']].corr()



# d['Cabin_first_letter_B_class_1'] = (d['Pclass']==1) & (d['Cabin_first_letter_B']==1)
# d['Cabin_first_letter_D_class_1'] = (d['Pclass']==1) & (d['Cabin_first_letter_D']==1)
# d['Cabin_first_letter_D_class_2'] = (d['Pclass']==2) & (d['Cabin_first_letter_D']==1)
# d['Cabin_first_letter_E_class_1'] = (d['Pclass']==1) & (d['Cabin_first_letter_E']==1)
# d['Cabin_first_letter_E_class_2'] = (d['Pclass']==2) & (d['Cabin_first_letter_E']==1)
# d['Cabin_first_letter_E_class_3'] = (d['Pclass']==3) & (d['Cabin_first_letter_E']==1)
# d['Cabin_first_letter_F_class_2'] = (d['Pclass']==2) & (d['Cabin_first_letter_F']==1)


# Cabin first letter did not help in CV or in public LB

# - order by ticket and find nannies


# tickets_children_without_parents = d[(d['Age']<=16) & (d['Parch']==0)]['Ticket'].values
# tickets_children_without_parents



# d[d['Ticket'].isin(tickets_children_without_parents)]


# No special survivors..

# From confusion matrix:
# - First class male, rather always with cabin filled and traveling alone
# - Third class females, without cabin,


# d['firstClassMaleWithCabinTravellingAlone'] = d.apply(lambda d: d['Sex_female']==False and d['Pclass']==1 and d['Cabin_filled']==True and d['family_isAlone']==True, axis=1)
# d['thirdClassFemaleWithoutCabin'] = d.apply(lambda d: d['Sex_female']==True and d['Pclass']==3 and d['Cabin_filled']==False, axis=1)


# No improvement on CV on public score

# - avg fare par family member


# d['farePerPerson'] = d['Fare'] / d['familySize']


# Correlation of fare per person to target is 22% while fare log is 33%

# Final set of features


features = ['Age', 'Pclass', 'Fare_log', 'Sex_female', 'Embarked_C', 'Embarked_Q', 'Cabin_filled'
           ] + ['family_isAlone', 'family_big'
           ] + ['isMr_NameLength', 'isMrs_NameLength', 'isMaster_NameLength', 'isMiss_NameLength'
           ] + ['isMr_LongName_Pclass_1', 'isMaster_LongName_Pclass_3']
X = d[features]



corr = d[features + ['Survived']].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, ax=ax, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()
corr


# ## Model selection

# ### Hyperparameters tuning 


grid_search_CV = GridSearchCV(RandomForestClassifier(n_estimators=100, class_weight='balanced'), {
    'criterion': ['gini', 'entropy'],
    'max_features': [3,4,5],
    'min_samples_split': [7, 10, 12, 20]
#     'max_depth': [3,4, 5]
}, n_jobs=7, cv=StratifiedKFold(n_splits=4, shuffle=True), verbose=10)

grid_search_CV.fit(X, y)
best_rf = grid_search_CV.best_estimator_
grid_search_CV.best_params_, grid_search_CV.best_score_
# 0.8372
# {'criterion': 'entropy', 'max_features': 3, 'min_samples_split': 20},



import graphviz
from sklearn.tree import export_graphviz
dot_data = export_graphviz(best_rf.estimators_[2], out_file=None, feature_names=features, filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph



grid_search_CV = GridSearchCV(XGBClassifier(n_estimators=400, max_depth=3, min_child_weight=2, gamma=0.1, reg_alpha=10, subsample=0.9, colsample_bytree=0.9, objective='binary:logistic'), {
    'max_depth': [4,5,6],
    'min_child_weight': [3, 4, 6],
    'gamma':[0.1, 1, 10],
    'reg_alpha':[1e-5, 1e-4, 1e-2, 0.1, 1]
}, n_jobs=7, cv=StratifiedKFold(n_splits=4, shuffle=True), verbose=10)

grid_search_CV.fit(X, y)
best_xgb = grid_search_CV.best_estimator_
grid_search_CV.best_params_, grid_search_CV.best_score_
# 0.840
# {'gamma': 1, 'max_depth': 5, 'min_child_weight': 6, 'reg_alpha': 0.1},



from sklearn.linear_model import LogisticRegression

grid_search_CV = GridSearchCV(LogisticRegression(), {
    'C': np.logspace(0.1, 10, num=5),
    'class_weight': [None, 'balanced']
}, n_jobs=7, cv=StratifiedKFold(n_splits=4, shuffle=True), verbose=10)

grid_search_CV.fit(d[['Age', 'Pclass', 'Fare', 'Sex_female', 'Embarked_C', 'Embarked_Q'] + ['family_isAlone', 'family_big'] +  ['isMr_NameLength', 'isMrs_NameLength', 'isMaster_NameLength', 'isMiss_NameLength'] + ['isMr_LongName_Pclass_1', 'isMaster_LongName_Pclass_3']], y)
best_logit = grid_search_CV.best_estimator_
grid_search_CV.best_params_, grid_search_CV.best_score_
# 0.8170
# {'C': 375.8374042884443, 'class_weight': 'balanced'}



from sklearn.ensemble import ExtraTreesClassifier

grid_search_CV = GridSearchCV(ExtraTreesClassifier(n_estimators=70), { ## TODO increase estimators??
    'max_depth': [4,5,6,8,10, 15, 20],
    'min_samples_split': [3, 4, 6, 8, 10],
}, n_jobs=7, cv=StratifiedKFold(n_splits=4, shuffle=True), verbose=10)

grid_search_CV.fit(X, y)
best_xtrees = grid_search_CV.best_estimator_
grid_search_CV.best_params_, grid_search_CV.best_score_
# 0.8428
# {'max_depth': 10, 'min_samples_split': 4}


# WARN: it takes time, like 10 minutes to run CV on one hyperparams set on that SVC


# from sklearn.svm import SVC

# grid_search_CV = GridSearchCV(SVC(kernel='linear', probability=True, degree=2), {
#     #'kernel': ['linear', 'poly'],
#     'kernel': ['poly'],
# #     'C': [1, 10, 100]
#      'C': [10]
# }, n_jobs=7, cv=StratifiedKFold(n_splits=4, shuffle=True), verbose=10)

# grid_search_CV.fit(X, y)
# best_svc = grid_search_CV.best_estimator_
# grid_search_CV.best_params_, grid_search_CV.best_score_
# # ({'C': 10, 'kernel': 'poly'}, 0.8283)



from sklearn.svm import SVC
best_svc = SVC(kernel='poly', probability=True, degree=2, C=10)
best_svc.fit(X, y)


# ### Learning curves
# Suspicion is we still have overfit problem

# Check for just RF, to start:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.05, 1.0, 6)):
    """
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()



plot_learning_curve(RandomForestClassifier(n_jobs=-1), "Learning Curves for RF", X, y, ylim=(0.7, 1.01), n_jobs=4)



# even more overfit??
plot_learning_curve(RandomForestClassifier(n_jobs=-1, n_estimators=50), "Learning Curves for RF", X, y, ylim=(0.7, 1.01), n_jobs=4)



plot_learning_curve(RandomForestClassifier(n_jobs=-1, n_estimators=50, criterion='gini', max_features=3, min_samples_split=10, max_depth=4), "Learning Curves for RF", X, y, ylim=(0.7, 1.01), n_jobs=4)



plot_learning_curve(best_rf, "Learning Curves for RF", X, y, ylim=(0.7, 1.01), n_jobs=4)



plot_learning_curve(best_logit, "Learning Curves for Logit", X, y, ylim=(0.7, 1.01), n_jobs=4)



# takes much time to fit SVC to all that sizes of test sets, and to CV it.
# plot_learning_curve(best_svc, "Learning Curves for SVC", X, y, ylim=(0.7, 1.01), n_jobs=4)


# Learning curve for SVC:
# 
# !['tSNE '](img/learning_curve_SVC_poly_degree_2_C_10.png)

# ## Validation curves


from sklearn.model_selection import validation_curve

def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None,
                        n_jobs=1):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(param_range, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(param_range, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()



best_rf



plot_validation_curve(best_rf, "Validation Curve for best RF", X, y, "min_samples_split", [int(v) for v in np.logspace(2, 6, base=1.7, num=6)], cv=StratifiedKFold(n_splits=4, shuffle=True), n_jobs=6)


# ## Ensembling

# #### Voting


# from sklearn.ensemble import VotingClassifier
# ensemble = VotingClassifier(estimators=[('rf', best_rf), ('xgb', best_xgb), ('logit', best_logit), ('xtrees', best_xtrees), ('svc', best_svc)], voting='hard')
# grid_search_CV = GridSearchCV(ensemble, {
#     'weights': [None, [2,2,1, 2, 1.5], [3,4,1, 2, 1.5]],
#     'voting': ['hard']
# }, n_jobs=-1, cv=StratifiedKFold(n_splits=4, shuffle=True), verbose=10)

# grid_search_CV.fit(X, y)
# best_voting = grid_search_CV.best_estimator_
# grid_search_CV.best_params_, grid_search_CV.best_score_
# ({'voting': 'hard', 'weights': [2, 2, 1, 2, 1.5]}, 0.82828282828282829)


# #### Stacking


from mlxtend.classifier import StackingCVClassifier
sclf = StackingCVClassifier(classifiers=[best_rf, best_xgb, best_logit, best_xtrees, best_svc], meta_classifier=LogisticRegression(), use_probas=True, cv=3, verbose=2)



cross_validate(sclf, X.values, y.values, cv=StratifiedKFold(n_splits=4, shuffle=True), n_jobs=-1)
# 'test_score': array([ 0.83035714,  0.83856502,  0.84234234,  0.82882883]),
# 'test_score': array([ 0.82589286,  0.86098655,  0.85135135,  0.80630631]),



sclf.fit(X.values, y.values)



# takes much time, like 30 minutes
# plot_learning_curve(sclf, "Learning Curves for ensemble of 2nd level - Stacker", X.values, y.values, ylim=(0.7, 1.01), n_jobs=4)


# Learning curve for Ensemble - Stacker of all 5 best classifiers:
# 
# !['Learning curve'](img/learning_curve_ensemble_stacker_all_5_best_classifiers.png)

# Testing out other combinations of base classifiers into for stackings:


# from mlxtend.classifier import StackingCVClassifier
# sclf = StackingCVClassifier(classifiers=[best_rf, best_xgb, best_xtrees, best_svc], meta_classifier=LogisticRegression(), use_probas=True, cv=3, verbose=2)
# cross_validate(sclf, X.values, y.values, cv=StratifiedKFold(n_splits=4, shuffle=True), n_jobs=-1)


# Check correlations of base classifiers predictions


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

rf=RandomForestClassifier(n_estimators=100, class_weight='balanced', criterion='entropy', max_features=3, min_samples_split=20)
xgb=XGBClassifier(n_estimators=400, max_depth=5, min_child_weight=6, gamma=1, reg_alpha=0.1, subsample=0.9, colsample_bytree=0.9, objective='binary:logistic')
logit=LogisticRegression(C=375, class_weight='balanced')
xtrees=ExtraTreesClassifier(n_estimators=70, max_depth=10, min_samples_split=4)
svc=SVC(kernel='poly', C=10, probability=True, degree=2)

classifiers = [(rf, 'rf'), (xgb, 'xgb'), (logit, 'logit'), (xtrees, 'xtrees'), (svc, 'svc')]
predictions=pd.DataFrame()
scores=pd.DataFrame()

def predict(c, classifier_name):
    c.fit(X_train, y_train)
    y_preds_proba = c.predict_proba(X_test)
    predictions[classifier_name] = y_preds_proba[:,1]
    y_preds = c.predict(X_test)
    scores[classifier_name] = accuracy_score(y_test, y_preds)
    
for cls, cls_name in classifiers:
    predict(cls, cls_name)

scores



cls_corr = predictions.corr()
fig, ax = plt.subplots()
sns.heatmap(cls_corr, ax=ax, 
            xticklabels=[name for _,name in classifiers],
            yticklabels=[name for _,name in classifiers])
plt.show()
cls_corr


# ## Confusion matrix


from sklearn.metrics import confusion_matrix

X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X, y)

tmp_classifier = LogisticRegression(class_weight='balanced', C=300)
tmp_classifier.fit(X_train_tmp, y_train_tmp)
y_pred = tmp_classifier.predict(X_test_tmp)

confusion_matrix(y_test_tmp, y_pred)



X_test_tmp['tmp_pred'] = y_pred
X_test_tmp['tmp_true'] = y_test_tmp


# False negatives:


X_test_tmp[(X_test_tmp['tmp_pred']==0) & (X_test_tmp['tmp_true']==1)]


# People who actually survived, but the classifier could not see it, were __male__, from 1 and 3 class.

# False positives


X_test_tmp[(X_test_tmp['tmp_pred']==1) & (X_test_tmp['tmp_true']==0)]


# People who __did not survive__, but the classifier asserted they did, were:
# 
# 1. First class male, rather always with cabin filled and traveling alone
# 2. Third class females, without cabin, 

# Why classifier is so good at predicting class 2?? And much much worse for the other 2 classes?


d[d['Pclass']==2][['Age', 'Sex_female', 'Fare', 'Survived']].corr()



d[d['Pclass']==1][['Age', 'Sex_female', 'Fare', 'Survived']].corr()



d[d['Pclass']==3][['Age', 'Sex_female', 'Fare', 'Survived']].corr()


# ## Test data


d_test = pd.read_csv('datasets' + os.sep + 'test.csv')



submission = pd.DataFrame()
submission['PassengerId'] = d_test['PassengerId']
del d_test['PassengerId']



d_test.columns[d_test.isnull().any()]



for pclass in [1,2,3]:
    d_test.loc[(d_test['Pclass']==pclass) & (d_test['Fare'].isnull()), ['Fare']] = d_test[(d_test['Pclass']==pclass)]['Fare'].mean()
# default if not matched:
d_test['Fare'].fillna(d_test['Fare'].mean(), inplace=True)



d_test = pd.get_dummies(d_test, columns=['Sex', 'Embarked']);



d_test['isMr'] = d_test['Name'].str.contains('Mr\.')
d_test['isMrs'] = d_test['Name'].str.contains('Mrs\.') # married woman
d_test['isMaster'] = d_test['Name'].str.contains('Master\.')
d_test['isMiss'] = d_test['Name'].str.contains('Miss\.') | d_test['Name'].str.contains('Mlle\.')
d_test['isMs'] = d_test['Name'].str.contains('Ms\.') # unmarried woman

d_test['isRev'] = d_test['Name'].str.contains('Rev\.') # clergy
d_test['isDr'] = d_test['Name'].str.contains('Dr\.') # rather as academic grade
d_test['isNoble'] = d_test['Name'].str.contains('Lady\.') | d_test['Name'].str.contains('Mme\.') | d_test['Name'].str.contains('Sir\.') | d_test['Name'].str.contains('Countess\.') | d_test['Name'].str.contains('Don\.') |  d_test['Name'].str.contains('Dona\.') | d_test['Name'].str.contains('Jonkheer\.')
d_test['isMilitary'] = d_test['Name'].str.contains('Major\.') | d_test['Name'].str.contains('Col\.') | d_test['Name'].str.contains('Capt\.')


assert d_test[~d_test['isMr'] & ~d_test['isMrs'] & ~d_test['isMaster'] & ~d_test['isMiss'] & ~d_test['isMs'] & ~d_test['isRev'] & ~d_test['isDr'] & ~d_test['isNoble'] & ~d_test['isMilitary']].empty



d_test['familySize'] = d_test['SibSp'] + d_test['Parch'] + 1
d_test['family_isAlone'] = d_test['familySize'] == 1 
d_test['family_big'] = d_test['familySize'] > 4 



d_test['Fare_log'] = np.log2(d_test['Fare'] + 1)



for f in ['isMr', 'isMrs', 'isMaster', 'isMiss']:
    print(f, d_test[d_test[f]==True]['Age'].mean(), d_test[d_test[f]==True].shape[0])



for pclass in [1,2,3]:
    for title in ['isMr', 'isMrs', 'isMaster', 'isMiss']:
        d_test.loc[(d_test[title]==True) & (d_test['Pclass']==pclass) & (d_test['Age'].isnull()), ['Age']] = d_test[(d_test[title]==True) & (d_test['Pclass']==pclass)]['Age'].mean()
# default
d_test['Age'].fillna(d_test['Age'].mean(), inplace=True)



d_test['Cabin_filled'] = d_test['Cabin'].notnull()



d_test['isMr_NameLength'] = d_test.apply(lambda r: len(r['Name'].split(' '))-1 if r['isMr']==True else 0, axis=1)
d_test['isMrs_NameLength'] = d_test.apply(lambda r: len(r['Name'].split(' '))-1 if r['isMrs']==True else 0, axis=1)
d_test['isMaster_NameLength'] = d_test.apply(lambda r: len(r['Name'].split(' '))-1 if r['isMaster']==True else 0, axis=1)
d_test['isMiss_NameLength'] = d_test.apply(lambda r: len(r['Name'].split(' '))-1 if r['isMiss']==True else 0, axis=1)
d_test['isMr_LongName_Pclass_1'] = d_test.apply(lambda r: len(r['Name'].split(' '))-1 >=4 if r['isMr']==True and r['Pclass']==1 else False, axis=1)
d_test['isMaster_LongName_Pclass_3'] = d_test.apply(lambda r: len(r['Name'].split(' '))-1 >=4 if r['isMaster']==True and r['Pclass']==3 else False, axis=1)



# d_test['Cabin_first_letter'] = d_test[d_test['Cabin'].notnull()]['Cabin'].str[:1]
# d_test = pd.get_dummies(d_test, columns=['Cabin_first_letter']);
# d_test['Cabin_first_letter_B_class_1'] = (d_test['Pclass']==1) & (d_test['Cabin_first_letter_B']==1)
# d_test['Cabin_first_letter_D_class_1'] = (d_test['Pclass']==1) & (d_test['Cabin_first_letter_D']==1)
# d_test['Cabin_first_letter_D_class_2'] = (d_test['Pclass']==2) & (d_test['Cabin_first_letter_D']==1)
# d_test['Cabin_first_letter_E_class_1'] = (d_test['Pclass']==1) & (d_test['Cabin_first_letter_E']==1)
# d_test['Cabin_first_letter_E_class_2'] = (d_test['Pclass']==2) & (d_test['Cabin_first_letter_E']==1)
# d_test['Cabin_first_letter_E_class_3'] = (d_test['Pclass']==3) & (d_test['Cabin_first_letter_E']==1)
# d_test['Cabin_first_letter_F_class_2'] = (d_test['Pclass']==2) & (d_test['Cabin_first_letter_F']==1)

# d_test['thirdClassFemaleWithoutCabin'] = d_test.apply(lambda d: d['Sex_female']==True and d['Pclass']==3 and d['Cabin_filled']==False, axis=1)



submission['Survived'] = sclf.predict(d_test[features].values)



date_time_stamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
submission.to_csv('submissions' + os.sep + 'submission_' + date_time_stamp + '.csv', float_format='%.4f', index=False)

