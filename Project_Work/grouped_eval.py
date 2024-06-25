from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_group_model(X_withid, X, y, pred_func):

    outer_cv = GroupKFold(n_splits = 5)
    inner_cv = GroupKFold(n_splits = 5)
    # tpr_best = 0

    results = {'pr_auc': [], 'roc_auc': [], 'precision': [], 'recall': [], 'fpr': [], 'tpr': [], 'y_true': [], 'y_pred': []}

    best_models = []

    for outer_train_idx, outer_test_idx in outer_cv.split(X, y, groups = X_withid['IDno']):
        X_outer_train, y_outer_train = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        X_outer_test, y_outer_test = X.iloc[outer_test_idx], y.iloc[outer_test_idx]

        inner_pr_auc_scores = []
        inner_roc_auc_scores = []
        best_inner_model = None
        best_inner_score = -np.inf

        for inner_train_idx, inner_test_idx in inner_cv.split(X_outer_train, y_outer_train, groups=X_withid['IDno'].iloc[outer_train_idx]):
            X_inner_train, y_inner_train = X_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_train_idx]
            X_inner_test, y_inner_test = X_outer_train.iloc[inner_test_idx], y_outer_train.iloc[inner_test_idx]

            y_pred_prob, inner_model = pred_func(X_inner_train, y_inner_train, X_inner_test)
    
            precision, recall, _ = precision_recall_curve(y_inner_test, y_pred_prob)
            pr_auc = auc(recall, precision)
            inner_pr_auc_scores.append(pr_auc)

            # fpr, tpr, _ = roc_curve(y_inner_test, y_pred_prob)
            roc_auc = roc_auc_score(y_inner_test, y_pred_prob)
            inner_roc_auc_scores.append(roc_auc)

            #if tpr > tpr_best:
            #    tpr_best = tpr
            #    best_vals = y_pred_vals

            if pr_auc > best_inner_score:
                best_inner_score = pr_auc
                best_inner_model = inner_model

        best_models.append(best_inner_model)

        y_pred_prob = best_inner_model.predict_proba(X_outer_test)[:, 1]

        # Compute PR AUC
        precision, recall, _ = precision_recall_curve(y_outer_test, y_pred_prob)
        pr_auc = auc(recall, precision)
        results['pr_auc'].append(pr_auc)

        # Compute ROC AUC
        fpr, tpr, _ = roc_curve(y_outer_test, y_pred_prob)
        roc_auc = roc_auc_score(y_outer_test, y_pred_prob)
        results['roc_auc'].append(roc_auc)

        results['precision'].append(precision)
        results['recall'].append(recall)
        results['fpr'].append(fpr)
        results['tpr'].append(tpr)
        results['y_true'].append(y_outer_test)
        results['y_pred'].append(y_pred_prob >= 0.5)

    return results, best_models

def print_auc_values_groups(model_name, results):
    print(f"Model: {model_name.upper()}")
    for i in range(len(results['roc_auc'])):
        print(f"Fold {i+1} - ROC AUC: {results['roc_auc'][i]:.4f}, PR AUC: {results['pr_auc'][i]:.4f}")
    print(f"Average ROC AUC: {np.mean(results['roc_auc']):.4f}")
    print(f"Average PR AUC: {np.mean(results['pr_auc']):.4f}")
    print()