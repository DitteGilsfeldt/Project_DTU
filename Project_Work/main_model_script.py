import pandas as pd
import glob
import os
import numpy as np
from datetime import datetime
from performance_eval import evaluate_model, plot_results, print_auc_values
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from grouped_eval import evaluate_group_model
import matplotlib.pyplot as plt

pre_op = pd.read_csv("L:/LovbeskyttetMapper01/HjertekirurgiDTU/Gruppe_1_DLMV/pre_operative_dataset.csv", index_col='IDno')
post_op = pd.read_csv("L:/LovbeskyttetMapper01/HjertekirurgiDTU/Gruppe_1_DLMV/post_operative_dataset.csv", index_col='IDno')

def standardize_scale(df, scale_cols):
    scaler = StandardScaler()
    for col in scale_cols:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    return df


def rf_pred(X_train, y_train, X_test):
    rf = RandomForestClassifier()

    rf_grid = {'max_depth': [80, 90, 100, 110],
    'max_features': [2,5,10,15,22],
    'n_estimators': [100, 200, 300]}
    
    grid_result = GridSearchCV(estimator = rf, param_grid = rf_grid, scoring='accuracy', cv=5)
    grid_result.fit(X_train,y_train)
    
    best_model = grid_result.best_estimator_ 

    rf_preds = best_model.predict_proba(X_test)[:,1]
    return rf_preds, best_model


def lr_pred(X_train, y_train, X_test, vals = False):
    lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1500)
    lr_grid = {
            'C': [100, 10, 1.0, 0.1, 0.01]}
    
    grid_result = GridSearchCV(estimator = lr, param_grid = lr_grid, scoring='accuracy', cv=5)
    grid_result.fit(X_train,y_train)
    
    best_model = grid_result.best_estimator_ 

    if vals == True:
        lr_vals = best_model.predict(X_test)
        lr_preds = best_model.predict_proba(X_test)[:,1]
        return lr_preds, lr_vals, best_model
    else:    
        lr_preds = best_model.predict_proba(X_test)[:,1]
        return lr_preds, best_model



def gbc_pred(X_train, y_train, X_test):
    
    gbc = GradientBoostingClassifier()
    gbc_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]}
    
    grid_result = GridSearchCV(estimator = gbc, param_grid = gbc_grid, scoring='accuracy', cv=5)
    grid_result.fit(X_train,y_train)
    
    best_model = grid_result.best_estimator_ 

    gbc_preds = best_model.predict_proba(X_test)[:,1]
    return gbc_preds, best_model


def mlp_pred(X_train, y_train, X_test):
   
    mlp = MLPClassifier(hidden_layer_sizes= (1,), max_iter=1500)
    mlp_grid = {
        'activation': ['relu', 'logistic'],
        'alpha': [0.0001, 0.001, 0.01]}

    grid_result = GridSearchCV(estimator=mlp, param_grid=mlp_grid, scoring='accuracy', cv=5)
    grid_result.fit(X_train, y_train)

    best_model = grid_result.best_estimator_

    mlp_preds = best_model.predict_proba(X_test)[:, 1]
    return mlp_preds, best_model


def filter_group(df, condition):
    return df[condition]



scales = ['age', 'Vægt','Hæmoglobin (preoperative)', 'Leukocytter (preoperative)', 'Trombocytter (preoperative)']
standardize_scale(pre_op, scale_cols=scales)
pre_op = pre_op.replace(True, 1).replace(False, 0)

pre_op = pre_op.dropna().replace(('Mand','Kvinde'),(1,0))

df = pre_op.reset_index()
X = df.loc[:, df.columns != 'Død inden for 1 år af operation']
X = X.loc[:, X.columns != 'IDno']
y = pre_op['Død inden for 1 år af operation']

results_lr, lr_model = evaluate_model(df, X, y, lr_pred, 'lr')
results_rf, rf_model = evaluate_model(df, X, y, rf_pred, 'rf')
results_gbc, gbc_model = evaluate_model(df, X, y, gbc_pred, 'gbc')
results_mlp, mlp_model = evaluate_model(df, X, y, mlp_pred, 'mlp')

plot_results('rf', results_rf)
plot_results('lr', results_lr)
plot_results('gbc', results_gbc)
plot_results('mlp', results_mlp)

print_auc_values('rf', results_rf)
print_auc_values('lr', results_lr)
print_auc_values('gbc', results_gbc)
print_auc_values('mlp', results_mlp)



men_lr = filter_group(df, df['sex'] == 1)
X_men_lr = men_lr.drop(['Død inden for 1 år af operation', 'IDno'], axis = 1)
y_men_lr = men_lr['Død inden for 1 år af operation']



results_men_lr, best_models_men_lr = evaluate_group_model(men_lr[['IDno']], X_men_lr, y_men_lr, lr_pred, 'lr')

y_true_men_lr = results_men_lr['y_true']
y_pred_men_lr = results_men_lr['y_pred']

print_auc_values('lr_men', results_men_lr)

females_lr = filter_group(df, df['sex'] == 0)
X_females_lr = females_lr.drop(['Død inden for 1 år af operation', 'IDno'], axis = 1)
y_females_lr = females_lr['Død inden for 1 år af operation']

results_females_lr, best_models_females_lr = evaluate_group_model(females_lr[['IDno']], X_females_lr, y_females_lr, lr_pred, 'lr')

y_true_females_lr = results_females_lr['y_true']
y_pred_females_lr = results_females_lr['y_pred']

print_auc_values('lr_women', results_females_lr)

old_lr = pre_op_original[pre_op_original['age'] > 60]

old_ID_lr = old_lr.index

old_standardized_lr = pre_op.loc[old_ID_lr]

X_old_lr = old_standardized_lr.drop(['Død inden for 1 år af operation'], axis = 1)
y_old_lr = old_standardized_lr['Død inden for 1 år af operation']

results_old_lr, best_models_old_lr = evaluate_group_model(old_standardized_lr.index.to_frame(), X_old_lr, y_old_lr, lr_pred, 'lr')

y_true_old_lr = results_old_lr['y_true']
y_pred_old_lr = results_old_lr['y_pred']

print_auc_values_groups('lr_old', results_old_lr)

young_lr = pre_op_original[pre_op_original['age'] <= 60]

young_ID_lr = young_lr.index

young_standardized_lr = pre_op.loc[young_ID_lr]

X_young_lr = young_standardized_lr.drop(['Død inden for 1 år af operation'], axis = 1)
y_young_lr = young_standardized_lr['Død inden for 1 år af operation']

results_young_lr, best_models_young_lr = evaluate_group_model(young_standardized_lr.index.to_frame(), X_young_lr, y_young_lr, lr_pred, 'lr')

print_auc_values_groups('lr_young', results_young_lr)

y_true_young_lr = results_young_lr['y_true']
y_pred_young_lr = results_young_lr['y_pred']


d = [0.2429, 0.1803, 0.2465, 0.1231]
bins = ['Men', 'Women', 'Above 60', 'Below 60']
plt.bar(bins, d, color = ['blue', 'orange', 'green', 'red'])
plt.ylabel('Average PR AUC')

for i, v in enumerate(d):
    plt.text(i, v, f'{v:2f}', ha = 'center', va = 'bottom')

plt.ylim(0, 0.5)
plt.show()

scales = ['age', 'Vægt','Hæmoglobin (preoperative)', 'Leukocytter (preoperative)', 'Trombocytter (preoperative)','Puls AVG(perioperativ)', 'Puls SD (perioperativ)', 'Saturation AVG(perioperativ)', 'Saturation SD(perioperativ)', 'Total blødning ml (perioperativ)', 'Total urin ml (perioperativ)','Hæmoglobin AVG (postoperative)', 'Hæmoglobin SD (postoperative)','Natrium AVG (postoperative)', 'Natrium SD (postoperative)','Kalium AVG (postoperative)', 'Kalium SD (postoperative)','Saturation AVG (postoperative)', 'Saturation SD (postoperative)','Puls AVG (postoperative)', 'Puls SD (postoperative)']
standardize_scale(post_op, scales)

post_op = post_op.replace(('Mand','Kvinde'),(1,0)).replace(('Ja','Nej'),(1,0)).replace((True,False),(1,0))

df_comp = post_op.reset_index().drop(columns='Død inden for 1 år af operation')
df_comp = df_comp.dropna()
X_comp = df_comp.loc[:, df_comp.columns != 'Nyresvigt']
X_comp = X_comp.loc[:, X_comp.columns != 'IDno']
y_comp = df_comp['Nyresvigt']


results_lr, lr_model = evaluate_group_model(df_comp, X_comp, y_comp, lr_pred, 'lr')
results_rf, rf_model = evaluate_group_model(df_comp, X_comp, y_comp, rf_pred, 'rf')
results_gbc, gbc_model = evaluate_group_model(df_comp, X_comp, y_comp, gbc_pred)
results_mlp, mlp_model = evaluate_group_model(df_comp, X_comp, y_comp, mlp_pred)

plot_results('lr', results_lr)
plot_results('lr', results_lr)

plot_results('rf', results_rf)
print_auc_values('rf', results_rf)

plot_results('gbc', results_gbc)
print_auc_values('gbc', results_gbc)

plot_results('mlp', results_mlp)
print_auc_values('mlp', results_mlp)