# Data handling
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

# Dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Visualization
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation


def plot_corr(df):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )


def plot_tsne(tnse_data, kmeans_labels):
    df_tsne = pd.DataFrame(tsne_data).rename({0: "x", 1: "y"}, axis=1)
    df_tsne["z"] = kmeans_labels
    sns.scatterplot(x=df_tsne.x, y=df_tsne.y, hue=df_tsne.z, palette="Set2")
    plt.show()


def prepare_pca(n_components, data, kmeans_labels):
    names = ["x", "y", "z"]
    matrix = PCA(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename(
        columns={i: names[i] for i in range(n_components)}, inplace=True
    )
    df_matrix["labels"] = kmeans_labels

    return df_matrix


def prepare_tsne(n_components, data, labels):
    names = ["x", "y", "z"]
    matrix = TSNE(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename(
        columns={i: names[i] for i in range(n_components)}, inplace=True
    )
    df_matrix["labels"] = labels

    return df_matrix


def plot_3d(df, name="labels"):
    iris = px.data.iris()
    fig = px.scatter_3d(df, x="x", y="y", z="z", color=name, opacity=0.5)

    fig.update_traces(marker=dict(size=3))
    fig.show()


def plot_animation(df, label_column, name):
    def update(num):
        ax.view_init(200, num)

    N = 360
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        df["x"],
        df["y"],
        df["z"],
        c=df[label_column],
        s=6,
        depthshade=True,
        cmap="Paired",
    )
    # ax.set_zlim(-30, 25)
    # ax.set_xlim(-20, 20)
    plt.tight_layout()
    # ani = animation.FuncAnimation(fig, update, N, blit=False, interval=50)
    # ani.save('{}.html'.format(name), writer='imagemagick')
    plt.show()
