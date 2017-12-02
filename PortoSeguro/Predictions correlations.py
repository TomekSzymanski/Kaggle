


import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns



gb_preds = pd.read_csv('tmp' + os.sep + 'validation_set_preditions_gb_prev.csv')
rf_preds = pd.read_csv('tmp' + os.sep + 'validation_set_preditions_rf.csv')
xtrees_preds = pd.read_csv('tmp' + os.sep + 'validation_set_preditions_xtrees.csv')
xgb_preds = pd.read_csv('tmp' + os.sep + 'validation_set_preditions_xgb.csv')



merged = pd.DataFrame()
merged['gb_preds'] = gb_preds
merged['rf_preds'] = rf_preds
merged['xtrees_preds'] = xtrees_preds
merged['xgb_preds'] = xgb_preds



merged.head()



for p in ['gb_preds', 'rf_preds', 'xtrees_preds', 'xgb_preds']:
    plt.hist(merged[p], normed=True, bins=40)
    plt.title(p)
    plt.show()


# ## Correlations


for p in ['gb_preds', 'rf_preds', 'xtrees_preds', 'xgb_preds']:
    merged[p+'_ranked'] = merged[p].rank(pct=True, method='dense')



sns.heatmap(merged[['gb_preds_ranked', 'rf_preds_ranked', 'xtrees_preds_ranked', 'xgb_preds_ranked']].corr(), annot=True, cmap="YlGnBu")
plt.show()



merged.head()



for p in ['gb_preds', 'rf_preds', 'xtrees_preds', 'xgb_preds']:
    plt.hist(merged[p+'_ranked'], normed=True, bins=40)
    plt.title(p)
    plt.show()



from sklearn.metrics import roc_curve, auc

def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) -1
    return g

def gini_norm(y, pred):
    normed = gini(y, pred) / gini(y,y)
    return normed



y_test = pd.read_csv('tmp' + os.sep + 'test_set_targets.csv')



for p in ['gb_preds', 'rf_preds', 'xtrees_preds', 'xgb_preds']:
    print('{}: {:.4f}'.format(p, gini_norm(y_test, merged[p])))



averaged = merged[['gb_preds_ranked', 'rf_preds_ranked', 'xtrees_preds_ranked', 'xgb_preds_ranked']].mean(axis=1)
gini_norm(y_test, averaged)



averaged = merged[['rf_preds_ranked', 'gb_preds', 'xgb_preds']].mean(axis=1)
gini_norm(y_test, averaged)



import numpy as np

y_pred_xgb = merged['xgb_preds_ranked']
y_pred_rf = merged['gb_preds_ranked']
y_pred_gb = merged['gb_preds_ranked']

best_single_classifer_score = 0.2838

weights_xgb = np.arange(0.6, 1, 0.025)
max_improvement_over_best = 0
w_xgb_argmax = 0
w_rf_argmax = 0
w_gb_argmax = 0
for weight_xgb in weights_xgb:
    parts = np.arange(0.1, 1, 0.025)
    for weight_rf in [(1 - weight_xgb) * part for part in parts ]:
        parts2 = np.arange(0.1, 1, 0.025)
        weight_gb = 1 - (weight_xgb + weight_rf)
        averaged_preds = weight_xgb * y_pred_xgb + weight_rf * y_pred_rf + weight_gb * y_pred_gb
        g = gini_norm(y_test, averaged_preds)
        better_by = g - best_single_classifer_score
        if (better_by > max_improvement_over_best):
            max_improvement_over_best = better_by
            w_xgb_argmax = weight_xgb
            w_rf_argmax = weight_rf
            w_gb_argmax = weight_gb
            print('Best improvement: ' + str(max_improvement_over_best), w_xgb_argmax, w_rf_argmax, w_gb_argmax)



import numpy as np

y_pred_xgb = merged['xgb_preds_ranked']
y_pred_rf = merged['rf_preds_ranked']
y_pred_gb = merged['gb_preds_ranked']

best_single_classifer_score = 0.2838

weights_xgb = np.arange(0.8, 1, 0.02)
max_improvement_over_best = 0
w_xgb_argmax = 0
w_rf_argmax = 0
w_gb_argmax = 0
for weight_xgb in weights_xgb:
    parts = np.arange(0.1, 1, 0.02)
    for weight_rf in [(1 - weight_xgb) * part for part in parts ]:
        parts2 = np.arange(0.1, 1, 0.025)
        weight_gb = 1 - (weight_xgb + weight_rf)
        averaged_preds = weight_xgb * y_pred_xgb + weight_rf * y_pred_rf + weight_gb * y_pred_gb
        g = gini_norm(y_test, averaged_preds)
        better_by = g - best_single_classifer_score
        if (better_by > max_improvement_over_best):
            max_improvement_over_best = better_by
            w_xgb_argmax = weight_xgb
            w_rf_argmax = weight_rf
            w_gb_argmax = weight_gb
            print('Best improvement: ' + str(max_improvement_over_best), w_xgb_argmax, w_rf_argmax, w_gb_argmax)



import numpy as np

y_pred_xgb = merged['xgb_preds_ranked']
y_pred_rf = merged['rf_preds_ranked']
y_pred_gb = merged['gb_preds_ranked']
y_pred_xtrees = merged['xtrees_preds_ranked']

best_single_classifer_score = 0.2838
weights_xgb = np.arange(0.85, 1, 0.02)
max_improvement_over_best = 0
w_xgb_argmax = 0
w_gb_argmax = 0
w_rf_argmax = 0
w_xtrees_argmax = 0
for weight_xgb in weights_xgb:
    parts = np.arange(0.1, 1, 0.025)
    for weight_rf in [(1 - weight_xgb) * part for part in parts ]:
        parts2 = np.arange(0.1, 1, 0.025)
        for weight_gb in [(1 - (weight_xgb + weight_rf)) * part for part in parts2 ]:
            weight_xtrees = 1 - (weight_xgb + weight_rf + weight_gb)
            averaged_preds = weight_xgb * y_pred_xgb + weight_rf * y_pred_rf + weight_gb * y_pred_gb + weight_xtrees * y_pred_xtrees
            g = gini_norm(y_test, averaged_preds)
            better_by = g - best_single_classifer_score
            if (better_by > max_improvement_over_best):
                max_improvement_over_best = better_by
                w_xgb_argmax = weight_xgb
                w_gb_argmax = weight_gb
                w_rf_argmax = weight_rf
                w_xtrees_argmax = weight_xtrees
                print('Best improvement: ' + str(max_improvement_over_best), w_xgb_argmax, w_gb_argmax, w_rf_argmax, w_xtrees_argmax)

