import numpy as np

def compute_cosin_distance(Q, feats):
    """
    feats and Q: L2-normalize, n*d
    """
    dists = np.dot(Q, feats.T)
    idxs = np.argsort(dists)[::-1]
    return dists,idxs

def compute_euclidean_distance(query, feats, names, k = None):
    if k is None:
        k = len(feats)

    dists = ((query - feats)**2).sum(axis=1)
    idx = np.argsort(dists)
    dists = dists[idx]
    rank_names = [names[k] for k in idx]
    return (idx[:k], dists[:k], rank_names)