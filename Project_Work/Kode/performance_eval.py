from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(X_withid, X, y, pred_func, model_name):

    outer_cv = GroupKFold(n_splits = 5)
    inner_cv = GroupKFold(n_splits = 5)

    results = {
    'lr': {'pr_auc': [], 'roc_auc': [], 'precision': [], 'recall': [], 'fpr': [], 'tpr': []},
    'rf': {'pr_auc': [], 'roc_auc': [], 'precision': [], 'recall': [], 'fpr': [], 'tpr': []},
    'gbc': {'pr_auc': [], 'roc_auc': [], 'precision': [], 'recall': [], 'fpr': [], 'tpr': []},
    'mlp': {'pr_auc': [], 'roc_auc': [], 'precision': [], 'recall': [], 'fpr': [], 'tpr': []}}

    for outer_train_idx, outer_test_idx in outer_cv.split(X, y, groups = X_withid['IDno']):
        X_outer_train, y_outer_train = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        X_outer_test, y_outer_test = X.iloc[outer_test_idx], y.iloc[outer_test_idx]

        inner_pr_auc_scores = []
        inner_roc_auc_scores = []

        for inner_train_idx, inner_test_idx in inner_cv.split(X_outer_train, y_outer_train, groups=X_withid['IDno'].iloc[outer_train_idx]):
            X_inner_train, y_inner_train = X_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_train_idx]
            X_inner_test, y_inner_test = X_outer_train.iloc[inner_test_idx], y_outer_train.iloc[inner_test_idx]

            y_pred_prob, best_model = pred_func(X_inner_train, y_inner_train, X_inner_test)
    
            precision, recall, _ = precision_recall_curve(y_inner_test, y_pred_prob)
            pr_auc = auc(recall, precision)
            inner_pr_auc_scores.append(pr_auc)

            fpr, tpr, _ = roc_curve(y_inner_test, y_pred_prob)
            roc_auc = roc_auc_score(y_inner_test, y_pred_prob)
            inner_roc_auc_scores.append(roc_auc)

        y_pred_prob, _ = pred_func(X_outer_train, y_outer_train, X_outer_test)

        # Compute PR AUC
        precision, recall, _ = precision_recall_curve(y_outer_test, y_pred_prob)
        pr_auc = auc(recall, precision)
        results[model_name]['pr_auc'].append(pr_auc)

        # Compute ROC AUC
        fpr, tpr, _ = roc_curve(y_outer_test, y_pred_prob)
        roc_auc = roc_auc_score(y_outer_test, y_pred_prob)
        results[model_name]['roc_auc'].append(roc_auc)

        results[model_name]['precision'].append(precision)
        results[model_name]['recall'].append(recall)
        results[model_name]['fpr'].append(fpr)
        results[model_name]['tpr'].append(tpr)
    return results, best_model


def plot_results(model_name, results):
    plt.figure(figsize=(14, 6))

    # Plot PR curves
    plt.subplot(1, 2, 1)
    for i in range(len(results['precision'])):
        plt.plot(results['recall'][i], results['precision'][i], alpha=1, label=f'Outer fold {i+1}', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR curve for {model_name.upper()}')
    plt.legend()

    # Plot ROC curves
    plt.subplot(1, 2, 2)
    for i in range(len(results['fpr'])):
        plt.plot(results['fpr'][i], results['tpr'][i], alpha=1, label=f'Outer fold {i+1}', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve for {model_name.upper()}')
    plt.legend()

    plt.tight_layout()
    plt.show()

def print_auc_values(model_name, results):
    print(f"Model: {model_name.upper()}")
    for i in range(len(results['roc_auc'])):
        print(f"Fold {i+1} - ROC AUC: {results['roc_auc'][i]:.4f}, PR AUC: {results['pr_auc'][i]:.4f}")
    print(f"Average ROC AUC: {np.mean(results['roc_auc']):.4f}")
    print(f"Average PR AUC: {np.mean(results['pr_auc']):.4f}")
    print()