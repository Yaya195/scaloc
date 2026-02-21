import numpy as np
from sklearn.cluster import KMeans, DBSCAN


def run_kmeans(x: np.ndarray, n_clusters: int) -> np.ndarray:
    return KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(x)


def run_dbscan(x: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
