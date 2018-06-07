import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    test_data = pd.read_csv("test.csv")
    train_data = pd.read_csv("train.csv")

    test_data["num"] = test_data["num"].replace([1, 2, 3, 4, 5, 6], 1)
    train_data["num"] = train_data["num"].replace([1, 2, 3, 4, 5, 6], 1)

    X_train = train_data.drop(['num'], axis=1)
    y_train = train_data['num']
    X_test = test_data.drop(['num'], axis=1)
    y_test = test_data['num']

    clf = RandomForestClassifier()
    parameters = {'n_estimators': [4, 6, 9],
                  'max_features': ['log2', 'sqrt', 'auto'],
                  'criterion': ['entropy', 'gini'],
                  'max_depth': [2, 3, 5, 10,15],
                  'min_samples_split': [2, 3, 5,7],
                  'min_samples_leaf': [1, 5, 8,11]}

    acc_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer,n_jobs=-1)
    grid_obj = grid_obj.fit(X_train, y_train)
    clf = grid_obj.best_estimator_
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(accuracy_score(y_test, predictions))

    all_data = pd.read_csv('data.csv')
    all_data["num"] = all_data["num"].replace([1, 2, 3, 4, 5, 6], 1)
    X_all = all_data.drop(['num'], axis=1)
    y_all = all_data['num']
    kf = KFold(2)
    outcomes = []
    fold = 0
    for train_index, test_index in kf.split(X=X_all):
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
        mean_outcome = np.mean(outcomes)
        print("Mean Accuracy: {0}".format(mean_outcome))
