import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from . import visualization
import string
import pandas as pd
import numpy as np
from kneed import KneeLocator


def elbow(df, normalize=False, k_range=range(2, 21), visualize=True):
    df_ = df.copy()
    if normalize:
        df_ = preprocessing.normalize(df)

    scores = [
        KMeans(n_clusters=i, random_state=10).fit(df_).inertia_
        for i in k_range
    ]

    if visualize:
        sns.lineplot(k_range, scores)
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.title("Inertia of k-Means versus number of clusters")
        plt.show()

    kn = KneeLocator(k_range, scores, curve="convex", direction="decreasing")
    # print(kn.elbow == kn.knee)
    return kn.elbow


def debug_dbscan(self, eps_arr=range(1, 10), debug=True):
    n_noise_arr = []
    n_clusters_arr = []
    for eps_ in eps_arr:
        db = DBSCAN(min_samples=10).fit(self.df)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        n_noise_arr.append(n_noise_)
        n_clusters_arr.append(n_clusters_)
        if n_clusters_ > 1 and debug == True:
            print(eps_)
            print(n_noise_)
            print(n_clusters_)
            print()
    plt.plot(eps_arr, n_clusters_arr)
    plt.show()

    plt.plot(eps_arr, n_noise_arr)
    plt.show()


class Analyzer:
    algorithm_args = {}
    clustered = None
    model = None
    predicted = None
    original = None

    def __init__(self, df):
        self.original = df.copy()
        self.df = df.copy()
        self.set_algorithm(algorithm="kmeans", algorithm_args={"k": 2})

    def set_algorithm(
        self, algorithm=["kmeans", "cosine_kmeans"], algorithm_args={}
    ):
        self.algorithm = algorithm
        self.algorithm_args = algorithm_args

    def train(self):
        if self.algorithm == "kmeans":
            self.model = KMeans(
                n_clusters=self.algorithm_args["n_clusters"], random_state=10
            )
            self.clustered = self.model.fit(self.df)
        elif self.algorithm == "cosine_kmeans":
            self.model = KMeans(
                n_clusters=self.algorithm_args["n_clusters"], random_state=10
            )

            df_normalized = preprocessing.normalize(self.df)
            self.clustered = self.model.fit(df_normalized)
        else:
            print("algorithm not supported")
            return

    def add_cluster_val(self, row, y_pred):
        row["cluster"] = str(string.ascii_lowercase[y_pred[row.name]])
        return row

    def predict_labels(self, idx, apply=False):
        if self.model is None:
            self.train()
        if self.predicted is None:
            self.df.reset_index(inplace=True)
            print(self.df.columns)
            id_df = self.df[[idx]]
            print(pd.__version__)
            self.df.drop(columns=[idx], inplace=True)
            self.predicted = self.model.predict(self.df)

            self.df = self.df.apply(
                self.add_cluster_val, axis=1, args=(self.predicted,)
            )
            self.df = self.df.join(id_df, how="inner")
            self.df.set_index(idx, inplace=True)
        return self.df[["cluster"]]

    def top_vars(self, idx, top=3, plot=True):
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(self.original),
            columns=self.original.columns,
            index=self.original.index,
        )
        df_scaled = df_scaled.join(self.predict_labels(idx), how="inner")

        df_ = pd.DataFrame(
            self.original,
            columns=self.original.columns,
            index=self.original.index,
        )
        df_ = df_.join(self.predict_labels(idx), how="inner")

        df_mean = df_scaled.groupby(["cluster"]).mean()

        # df_mean.set_index('cluster', inplace=True)
        results = pd.DataFrame(columns=["Variable", "Var"])
        print(len(df_mean.columns))
        df_mean = df_mean.loc[:, ~df_mean.columns.duplicated()]
        print(len(df_mean.columns))

        for column in df_mean.columns[0:]:
            results.loc[len(results), :] = [column, np.std(df_mean[column])]

        df_mean.reset_index(inplace=True)
        selected_columns = list(
            results.sort_values(
                "Var",
                ascending=False,
            )
            .head(top)
            .Variable.values
        ) + ["cluster"]
        tidy = df_scaled[selected_columns].melt(id_vars="cluster")
        sns.barplot(x="cluster", y="value", hue="variable", data=tidy)
        plt.show()
        return results.sort_values(by="Var", ascending=False).head(top)

    def evaluate_silhouette_score(self, metric):  # euclidean or cosine
        if self.clustered == None:
            self.train()
        score = silhouette_score(
            self.df, self.clustered.labels_, metric=metric
        )
        print((self.algorithm + " - silhouette_score: {}").format(score))
        return score

    def pca(self, n=2):
        pca_df = visualization.prepare_pca(n, self.df, self.clustered.labels_)
        sns.scatterplot(
            x=pca_df["x"], y=pca_df["y"], hue=pca_df["labels"], palette="Set1"
        )

    def tsne_3d(self, name=""):
        if name is None or name == "":
            name = self.algorithm
        tsne_3d_df = visualization.prepare_tsne(
            3, self.df, self.clustered.labels_
        )
        tsne_3d_df[self.algorithm] = self.clustered.labels_

        visualization.plot_animation(tsne_3d_df, self.algorithm, name)
