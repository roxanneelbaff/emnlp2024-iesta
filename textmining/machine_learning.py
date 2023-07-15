from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import pickle


def dummy_train_test(
    X_train, y_train, X_test, y_test, strategy="most_frequent"
):
    clf = DummyClassifier(strategy=strategy, random_state=1)
    """
    “stratified”: generates predictions by respecting the training set’s class distribution.

    “most_frequent”: always predicts the most frequent label in the training set.

    “prior”: always predicts the class that maximizes the class prior (like “most_frequent”) and predict_proba returns the class prior.

    “uniform”: generates predictions uniformly at random.

    “constant”: always predicts a constant label that is provided by the user. This is useful for metrics that evaluate a non-majority class
    """

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1_dic = {}

    f1_dic["macro"] = round(
        f1_score(y_pred=y_pred, y_true=y_test, average="macro"), 2
    )
    f1_dic["micro"] = round(
        f1_score(y_pred=y_pred, y_true=y_test, average="micro"), 2
    )

    classes = list(np.unique(y_train))
    for label in classes:
        f1_dic[label] = round(
            f1_score(
                y_pred=y_pred, y_true=y_test, average=None, labels=[label]
            )[0],
            2,
        )

    return f1_dic


### Outliers
def _get_upper_lower_df(train_df, lower_percentile=1, upper_percentile=99):
    boundaries = np.percentile(
        train_df, [lower_percentile, upper_percentile], axis=0
    )
    boundaries_df = pd.DataFrame(boundaries, columns=train_df.columns)
    return boundaries_df


def _apply_clip_col(col, upper_lower_df):
    if col.name in upper_lower_df.columns:
        col = np.clip(
            col,
            max(upper_lower_df[col.name].values),
            min(upper_lower_df[col.name].values),
        )

    return col


def clip_outliers(
    df_train, df_test=None, lower_percentile=1, upper_percentile=99
):
    print("getting only numeric features from the training set...")
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    numeric_df = df_train.select_dtypes(include=numerics)
    print(
        "There are {}  numeric features out of {}".format(
            str(len(numeric_df.columns)), str(len(df_train.columns))
        )
    )
    boundaries_df = _get_upper_lower_df(
        numeric_df, lower_percentile=1, upper_percentile=99
    )

    df_train = df_train.apply(_apply_clip_col, args=(boundaries_df,), axis=0)
    if df_test is not None:
        df_test = df_test.apply(_apply_clip_col, args=(boundaries_df,), axis=0)

    return df_train, df_test


def normalize(X_train, X_test, normalizing_method="standard"):
    if normalizing_method == "standard":
        print("Normalizing by using standard scaler...")
        scaler = StandardScaler(copy=True, with_mean=False)
        scaler.fit(X_train)
        X_train = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test = (
            pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
            if (X_test is not None)
            else None
        )
    elif normalizing_method == "log":
        X_train = np.log(X_train + 1)
        X_test = np.log(X_test + 1) if (X_test is not None) else None
    elif normalizing_method == "sqrt":
        X_train = np.sqrt(X_train + (2 / 3))
        X_test = np.sqrt(X_test + (2 / 3)) if (X_test is not None) else None
    elif normalizing_method == "minmax":
        scaler = MinMaxScaler(copy=True)
        scaler.fit(X_train)

        X_train = pd.DataFrame(
            scaler.transform(X_train), columns=X_train.columns
        )
        X_test = (
            pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
            if (X_test is not None)
            else None
        )
    else:
        print(
            "data is not normalized. method is not supported. Choose one of: standard, minmax, log, sqrt"
        )

    return X_train, X_test


def train_save(
    X_train, y_train, pkl_filename, algorithm="svm", details=False, params={}
):
    # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    print("running ", algorithm, " with params ", str(params))
    if algorithm == "svm":
        cw = params["class_weight"] if "class_weight" in params else None
        c = params["C"] if "C" in params else "balanced"
        clf = SVC(kernel="linear", class_weight=cw, C=c)
    elif algorithm == "randomforest":
        estimator = params["n_estimators"] if "n_estimators" in params else 1
        max_depth = params["max_depth"] if "max_depth" in params else 2
        clf = RandomForestClassifier(
            n_estimators=estimator, max_depth=max_depth, random_state=1
        )
    else:
        print(
            "algorithm not supported! valid values are: svm and randomforest"
        )
        return

    clf.fit(X_train, y_train)
    with open(pkl_filename, "wb") as f:
        pickle.dump(clf, f)


def train_test(
    X_train, y_train, X_test, y_test, algorithm="svm", details=False, params={}
):
    # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    if details:
        print("running ", algorithm, " with params ", str(params))
    if algorithm == "svm":
        cw = params["class_weight"] if "class_weight" in params else None
        c = params["C"] if "C" in params else "balanced"
        clf = SVC(kernel="linear", class_weight=cw, C=c)
    elif algorithm == "randomforest":
        estimator = params["n_estimators"] if "n_estimators" in params else 1
        max_depth = params["max_depth"] if "max_depth" in params else 2
        clf = RandomForestClassifier(
            n_estimators=estimator, max_depth=max_depth, random_state=1
        )
    else:
        print(
            "algorithm not supported! valid values are: svm and randomforest"
        )
        return

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    if details:
        print("Confusion Matrix:")

        print(confusion_matrix(y_test, y_pred))

        print()
        print("Accuracy: ", round(accuracy_score(y_test, y_pred), 2))
        print("Report:")
        print(classification_report(y_test, y_pred))
    f1_dic = {}

    f1_dic["macro"] = round(
        f1_score(y_pred=y_pred, y_true=y_test, average="macro"), 2
    )
    f1_dic["micro"] = round(
        f1_score(y_pred=y_pred, y_true=y_test, average="micro"), 2
    )

    f1_dic["accuracy"] = round(accuracy_score(y_test, y_pred), 2)
    # f1_dic['no-effect'] = round(f1_score(y_pred=y_pred, y_true=y_test, average=None, labels=['no-effect'])[0], 2)
    classes = list(np.unique(y_train))
    for label in classes:
        f1_dic[label] = round(
            f1_score(
                y_pred=y_pred, y_true=y_test, average=None, labels=[label]
            )[0],
            2,
        )

    return f1_dic


def svc_param_gridsearch(X, y, nfolds_or_division):
    Cs = [0.01, 0.1, 1, 10, 100]
    param_grid = {"C": Cs, "class_weight": ["balanced"]}  # , 'gamma' : gammas}
    grid_search = GridSearchCV(
        SVC(kernel="linear"),
        param_grid,
        cv=nfolds_or_division,
        n_jobs=2,
        scoring="f1_macro",
    )

    grid_search.fit(X, y)

    return grid_search.best_params_  # , grid_search.best_score_


def randomforest_param_gridsearch(X, y, nfolds_or_division):
    estimators = range(1, 41)
    param_grid = {"n_estimators": estimators}  # , 'gamma' : gammas}
    grid_search = GridSearchCV(
        RandomForestClassifier(n_jobs=-1, class_weight="balanced"),
        param_grid,
        n_jobs=-1,
        scoring="f1_macro",
    )

    grid_search.fit(X, y)

    return grid_search.best_params_  # , grid_search.best_score_
