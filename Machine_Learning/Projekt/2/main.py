import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LogisticRegressionCV, LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from scipy.stats import t
import seaborn as sns


def preprocess():
    # Load the dataset
    df = pd.read_csv("data.csv")

    df = df.drop(['row.names', 'chd'], axis=1)

    # Selecting the relevant features and target
    X = df[['sbp', 'ldl', 'obesity', 'age']]
    y = df['adiposity']

    colx = {}

    for a in X.columns:
        mean = X[a].mean()
        std = X[a].std()
        colx[a] = (X[a] - mean) / std

    sDX = pd.DataFrame(colx)
    coefficients = pd.DataFrame(index=lambdas, columns=X.columns)
    return sDX, y, X, coefficients


def cross_val(sDX, y, lambdas, kFolds, coefficients_df):
    kf = KFold(n_splits=kFolds, shuffle=True)
    test_error_matrix = np.zeros((kFolds, len(lambdas)))
    train_error_matrix = np.zeros((kFolds, len(lambdas)))

    # Create a matrix to store the fold-wise coefficients for averaging later
    fold_coefficients = np.zeros((kFolds, len(lambdas), sDX.shape[1]))

    fold_id = 0
    for train_i, test_i in kf.split(sDX):
        X_train, X_test = sDX.iloc[train_i], sDX.iloc[test_i]
        y_train, y_test = y.iloc[train_i], y.iloc[test_i]

        for i, l in enumerate(lambdas):
            model = Ridge(alpha=l)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_error_matrix[fold_id, i] = mean_squared_error(y_train, y_train_pred)
            test_error_matrix[fold_id, i] = mean_squared_error(y_test, y_test_pred)

            # Store the coefficients for this fold and lambda
            fold_coefficients[fold_id, i, :] = model.coef_
            print("Lambda: ", l, "coefficients: ", model.coef_,  "intercept: ", model.intercept_)

        fold_id += 1

    # Now that we've collected all fold-wise coefficients, we can average them
    avg_coefficients = fold_coefficients.mean(axis=0)

    # Update the coefficients DataFrame with the average values
    for i, l in enumerate(lambdas):
        coefficients_df.loc[l] = avg_coefficients[i, :]

    avg_train_error = train_error_matrix.mean(axis=0)
    avg_test_error = test_error_matrix.mean(axis=0)

    return coefficients_df, avg_train_error, avg_test_error


def two_fold_cross_val (k_outer, k_inner, lambdas, hidden_layers, sDX, y):
    kf_outer = KFold(n_splits=k_outer, shuffle=True)
    kf_inner = KFold(n_splits=k_inner, shuffle=True)
    ridge_error_array_size = (k_inner, len(lambdas))
    ann_error_array_size = (k_inner, len(hidden_layers))

    ridge_gen_error_array_size = (len(lambdas))
    ridge_gen_errors = np.zeros(ridge_gen_error_array_size)

    ann_gen_error_array_size = (len(hidden_layers))
    ann_gen_errors = np.zeros(ann_gen_error_array_size)

    opt_ridge_error = []
    opt_ann_error = []
    baseline_test_error = []

    opt_lambda_list = []
    opt_h_list = []

    input_dim = len(sDX.columns)

    for a, (train_outer, test_outer) in enumerate(kf_outer.split(sDX)):
        D_par = sDX.iloc[train_outer]
        D_test = sDX.iloc[test_outer]
        y_par = y.iloc[train_outer]
        y_test = y.iloc[test_outer]
        par_length = len(D_par)

        for b, (train_inner, val_inner) in enumerate(kf_inner.split(D_par)):
            D_train = D_par.iloc[train_inner]
            D_train_tensor = tf.convert_to_tensor(D_train)
            D_val = D_par.iloc[val_inner]
            y_train = y_par.iloc[train_inner]
            y_train_tensor = tf.convert_to_tensor(y_train)
            y_val = y_par.iloc[val_inner]
            val_length = len(D_val)

            ridge_val_error = np.zeros(ridge_error_array_size)
            ann_val_error = np.zeros(ann_error_array_size)


            for l, la in enumerate(lambdas):
                ridge_model = Ridge(alpha=la)
                ridge_model.fit(D_train, y_train)

                ridge_val_error[b, l] = (val_length/par_length)*mean_squared_error(ridge_model.predict(D_val), y_val)

            for id, h in enumerate(hidden_layers):
                ann_model = create_regression_ANN(h, input_dim, 'linear')
                ann_model.fit(D_train_tensor, y_train_tensor, batch_size=32, epochs=150, verbose=0)
                ann_val_error[b, id] = \
                    (val_length/par_length)*mean_squared_error(ann_model.predict(D_val), y_val)

        for l in range (len(lambdas)):
            ridge_gen_errors[l] = sum(ridge_val_error[:, l])

        for h in range (len(hidden_layers)):
            ann_gen_errors[h] = sum(ann_val_error[:, h])

        print("ridge gen: ", ridge_gen_errors)
        print("ann gen: ", ann_gen_errors)

        optimal_ridge_lambda = lambdas[np.argmin(ridge_gen_errors)]
        opt_lambda_list.append(optimal_ridge_lambda)
        optimal_ann_h = hidden_layers[np.argmin(ann_gen_errors)]
        opt_h_list.append(optimal_ann_h)
        baseline = np.mean(y_par)
        print(baseline)
        print(len(y_par))
        print(len(np.repeat(baseline, len(y_test))))

        opt_ridge = Ridge(alpha=optimal_ridge_lambda)
        opt_ridge.fit(D_par, y_par)
        opt_ridge_error.append(mean_squared_error(opt_ridge.predict(D_test), y_test))

        opt_ann = create_regression_ANN(optimal_ann_h, input_dim, 'linear')
        opt_ann.fit(D_par, y_par, batch_size=16, epochs=100, verbose=0)
        opt_ann_error.append(mean_squared_error(opt_ann.predict(D_test), y_test))

        baseline_test_error.append(mean_squared_error(y_test, np.repeat(baseline, len(test_outer))))

        print(opt_lambda_list)
        print(opt_ridge_error)

        print(opt_h_list)
        print(opt_ann_error)

        print(baseline_test_error)

    # analyzing the models pairwise

    # Ridge vs ANN
    ridge_ann = statistical_analysis(np.array(opt_ridge_error) - np.array(opt_ann_error))
    ridge_baseline = statistical_analysis(np.array(opt_ridge_error) - np.array(baseline_test_error))
    ann_baseline = statistical_analysis(np.array(opt_ann_error) - np.array(baseline_test_error))

    print("Ridge vs ANN: ", ridge_ann[0], ridge_ann[1])

    print("Ridge vs Baseline: ", ridge_baseline[0], ridge_baseline[1])

    print("ANN vs Baseline: ", ann_baseline[0], ann_baseline[1])

    return opt_ridge_error, opt_ann_error


def statistical_analysis(data, alpha=0.05):
    n = len(data)
    mu = np.mean(data)
    s = np.std(data, ddof=1)  # Sample standard deviation

    # Confidence interval
    ci = t.interval(1-alpha, n-1, loc=mu, scale=s/np.sqrt(n))

    # T-statistic
    t_stat = mu / (s/np.sqrt(n))

    # P-value
    if t_stat < 0:
        p = 2 * t.cdf(t_stat, df=n-1)
    else:
        p = 2 * (1 - t.cdf(t_stat, df=n-1))

    return ci, p


def create_regression_ANN (h, input_dim, act, output_size=1):
    model = Sequential()
    model.add(Dense(h, input_dim=input_dim, activation=act))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=Adam())
    return model


def visualize_coefficient_paths(lambdas, coefficients):
    plt.figure(figsize=(10, 6))
    for feature in coefficients.columns:
        plt.plot(lambdas, coefficients[feature], label=feature)
    plt.xscale('log')
    plt.xlabel('Regularization factor (λ)')
    plt.ylabel('Mean Coefficient Values')
    plt.title('Coefficient Paths')
    plt.legend()
    plt.show()


def visualize_errors(lambdas, avg_train_error, avg_test_error):
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, avg_train_error, label='Train error', linestyle='--')
    plt.plot(lambdas, avg_test_error, label='Validation error')
    plt.xscale('log')
    plt.xlabel('Regularization factor (λ)')
    plt.ylabel('Squared error (cross-validation)')
    plt.title('Model Error as a Function of Regularization Strength')
    plt.legend()
    # Add the vertical line for optimal lambda
    optimal_lambda = lambdas[np.argmin(avg_test_error)]
    plt.axvline(optimal_lambda, color='grey', linestyle='--', label=f'Optimal λ: {optimal_lambda}')
    plt.legend()
    plt.show()


def scatterPlot_data(sDX, y):
    data_for_pairplot = pd.concat([sDX, y], axis=1)

    # Now create the pairplot
    sns.pairplot(data_for_pairplot)
    plt.show()


def classify_data (y):
    # Calculate the mean of the 'adiposity' column
    adiposity_mean = y.mean()

    # Binarize the 'adiposity' column: 0 if below or equal to the mean, 1 if above
    y = (y > adiposity_mean).astype(int)
    return y


def logistic_regression_trial_run(sDX, y, lambdas, kFolds):
    kf = KFold(n_splits=kFolds, shuffle=True)
    error_matrix = np.zeros((kFolds, len(lambdas)))

    for i, (train_i, test_i) in enumerate(kf.split(sDX)):
        X_train, X_test = sDX.iloc[train_i], sDX.iloc[test_i]
        y_train, y_test = y.iloc[train_i], y.iloc[test_i]

        for j, l in enumerate(lambdas):
            model = LogisticRegression(C=1/l, penalty='l2', solver='liblinear', max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            incorrect_predictions = np.sum(y_pred != y_test)
            error_matrix[i, j] = incorrect_predictions / len(y_test)

    mean_errors = np.mean(error_matrix, axis=0)
    optimal_lambda_index = np.argmin(mean_errors)
    optimal_lambda = lambdas[optimal_lambda_index]
    print(f"Optimal Lambda: {optimal_lambda}")
    return optimal_lambda


def ANN_trial_run(sDX, y, h_values, kFolds):
    kf = KFold(n_splits=kFolds, shuffle=True)
    error_matrix = np.zeros((kFolds, len(h_values)))

    for fold_no, (train_i, test_i) in enumerate(kf.split(sDX)):
        X_train, X_test = sDX.iloc[train_i], sDX.iloc[test_i]
        y_train, y_test = y.iloc[train_i], y.iloc[test_i]

        for j, h in enumerate(h_values):
            model = create_regression_ANN(h, X_train.shape[1], 'relu')
            model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)
            y_pred = model.predict(X_test).flatten()
            y_pred_binary = (y_pred > 0.5).astype(int)
            incorrect_pred = np.sum(y_pred_binary != y_test)
            print("incorrect pred: ", incorrect_pred, "total: ", len(y_test))
            error_matrix[fold_no, j] = incorrect_pred / len(y_test)

    mean_errors = np.mean(error_matrix, axis=0)
    print(mean_errors)
    optimal_h_index = np.argmin(mean_errors)
    optimal_h = h_values[optimal_h_index]
    print(error_matrix)
    print(f"Optimal h: {optimal_h}")
    return optimal_h


def classification_two_level_cross_val(k_outer, k_inner, lambdas, hidden_layers, sDX, y):
    kf_outer = KFold(n_splits=k_outer, shuffle=True)
    kf_inner = KFold(n_splits=k_inner, shuffle=True)
    log_error_array_size = (k_inner, len(lambdas))
    ann_error_array_size = (k_inner, len(hidden_layers))

    log_gen_error_array_size = (len(lambdas))
    log_gen_errors = np.zeros(log_gen_error_array_size)

    ann_gen_error_array_size = (len(hidden_layers))
    ann_gen_errors = np.zeros(ann_gen_error_array_size)

    opt_log_error = []
    opt_ann_error = []
    baseline_test_error = []

    opt_lambda_list = []
    opt_h_list = []

    input_dim = len(sDX.columns)

    for a, (train_outer, test_outer) in enumerate(kf_outer.split(sDX)):
        D_par = sDX.iloc[train_outer]
        D_test = sDX.iloc[test_outer]
        y_par = y.iloc[train_outer]
        y_test = y.iloc[test_outer]
        par_length = len(D_par)

        for b, (train_inner, val_inner) in enumerate(kf_inner.split(D_par)):
            D_train = D_par.iloc[train_inner]
            D_train_tensor = tf.convert_to_tensor(D_train)
            D_val = D_par.iloc[val_inner]
            y_train = y_par.iloc[train_inner]
            y_train_tensor = tf.convert_to_tensor(y_train)
            y_val = y_par.iloc[val_inner]
            val_length = len(D_val)

            log_val_error = np.zeros(log_error_array_size)
            ann_val_error = np.zeros(ann_error_array_size)


            for l, la in enumerate(lambdas):
                log_model = LogisticRegression(C=1/la, penalty='l2', solver='liblinear', max_iter=1000)
                log_model.fit(D_train, y_train)
                y_pred = log_model.predict(D_val)
                log_incorrect_predictions = np.sum(y_pred != y_val)
                log_val_error[b, l] = log_incorrect_predictions/val_length

            for id, h in enumerate(hidden_layers):
                ann_model = create_regression_ANN(h, input_dim, 'relu')
                ann_model.fit(D_train_tensor, y_train_tensor, batch_size=32, epochs=150, verbose=0)
                y_pred = ann_model.predict(D_val).flatten()
                y_pred_binary = (y_pred > 0.5).astype(int)
                incorrect_pred = np.sum(y_pred_binary != y_val)
                ann_val_error[b, id] = incorrect_pred/val_length

        for l in range (len(lambdas)):
            log_gen_errors[l] = log_val_error[:, l].mean()

        for h in range (len(hidden_layers)):
            ann_gen_errors[h] = ann_val_error[:, h].mean()

        print("ridge gen: ", log_gen_errors)
        print("ann gen: ", ann_gen_errors)


        optimal_log_lambda = lambdas[np.argmin(log_gen_errors)]
        opt_lambda_list.append(optimal_log_lambda)
        optimal_ann_h = hidden_layers[np.argmin(ann_gen_errors)]
        opt_h_list.append(optimal_ann_h)
        count = np.bincount(y_par)
        baseline = np.argmax(count)
        print(baseline)
        print(len(y_par))
        print(len(np.repeat(baseline, len(y_test))))

        opt_log = LogisticRegression(C=1/optimal_log_lambda, penalty='l2', solver='liblinear', max_iter=1000)
        opt_log.fit(D_par, y_par)
        print("Lambda: ",optimal_log_lambda, "coeffs: ", opt_log.coef_, "intercept: ", opt_log.intercept_)
        opt_log_error.append(np.sum(opt_log.predict(D_test) != y_test)/len(y_test))

        opt_ann = create_regression_ANN(h, input_dim, 'relu')
        opt_ann.fit(D_par, y_par, batch_size=32, epochs=150, verbose=0)
        y_pred = opt_ann.predict(D_test).flatten()
        y_pred_binary = (y_pred > 0.5).astype(int)
        incorrect_pred = np.sum(y_pred_binary != y_test)
        opt_ann_error.append(incorrect_pred / len(y_test))

        baseline_test_error.append(np.sum(np.repeat(baseline, len(test_outer)) != y_test)/len(y_test))

    print(opt_lambda_list)
    print(opt_log_error)

    print(opt_h_list)
    print(opt_ann_error)

    print(baseline_test_error)

    # analyzing the models pairwise

    # Ridge vs ANN
    ridge_ann = statistical_analysis(np.array(opt_log_error) - np.array(opt_ann_error))
    ridge_baseline = statistical_analysis(np.array(opt_log_error) - np.array(baseline_test_error))
    ann_baseline = statistical_analysis(np.array(opt_ann_error) - np.array(baseline_test_error))

    print("Ridge vs ANN: ", ridge_ann[0], ridge_ann[1])

    print("Ridge vs Baseline: ", ridge_baseline[0], ridge_baseline[1])

    print("ANN vs Baseline: ", ann_baseline[0], ann_baseline[1])

    return opt_log_error, opt_ann_error



lambdas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
x_values = np.arange(len(lambdas))
h_values = [1, 4, 16, 64, 256, 1024, 4096]
kFolds = 10
sDX, y, X, coefficients = preprocess()

coefficients, avg_train_error, avg_test_error = cross_val(sDX, y, lambdas, kFolds, coefficients)


# scatterPlot_data(sDX, y)
# visualize_coefficient_paths(lambdas, pd.DataFrame(coefficients))
# visualize_errors(lambdas, avg_train_error, avg_test_error)
# ridge_errors, ann_errors = two_fold_cross_val(kFolds, kFolds, lambdas, h_values, sDX, y)

classified_y = classify_data(y)


#log_lambda = logistic_regression_trial_run(sDX, classified_y, lambdas, kFolds)
#log_h = ANN_trial_run(sDX, classified_y, h_values, kFolds)

log_errors, ann_errors = classification_two_level_cross_val(kFolds, kFolds, lambdas, h_values, sDX, classified_y)

