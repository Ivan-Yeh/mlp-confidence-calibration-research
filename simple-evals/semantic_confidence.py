import numpy as np
from collections import Counter

# ---------------------------------------------------------
# model_name="all-MiniLM-L6-v2"
# distance_threshold = 0.7
# lnll_lst = [(x)[1] for x in response_sample]
# response_list = [x[0] for x in response_sample]
# embeddings = SentenceTransformer(model_name).encode(response_list)
# clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average")
# labels = clustering.fit_predict(embeddings)
# ---------------------------------------------------------


def single_generation_confidence(lnll_lst, response_list, labels):
    return response_list[0], lnll_lst[0]


def empirical_semantic_confidence(lnll_lst, response_list, labels):
    counts = Counter(labels)
    opt_cluster, opt_conf = max([(int(cluster_id), count/sum(counts.values())) for cluster_id, count in counts.items()], key=lambda x: x[1])
    optimal_response = max([(response_list[i], lnll_lst[i]) for i, label in enumerate(labels) if label == opt_cluster], key=lambda x: x[1])[0]
    return optimal_response, opt_conf


def likelihood_based_semantic_confidence(lnll_lst, response_list, labels):
    clustered = []
    # for each cluster calculat the following:
    for c in np.unique(labels):
        sum_ci = sum([lnll_lst[i] for i in range(len(lnll_lst)) if labels[i] == c]) # s(Ci | x) = sum(LN-LL)
        clustered.append((int(c), sum_ci))
    total_lsc = sum([x[1] for x in clustered])
    lsc = [(c[0], c[1] / total_lsc) for c in clustered]
    
    opt_cluster, opt_conf = max(lsc, key=lambda x: x[1])
    optimal_response = max([(response_list[i], lnll_lst[i]) for i, label in enumerate(labels) if label == opt_cluster], key=lambda x: x[1])[0]

    return optimal_response, opt_conf


def mean_likelihood_based_semantic_confidence(lnll_lst, response_list, labels):
    clustered = []
    # for each cluster calculate s_bar(Ci | x):
    for c in np.unique(labels):
        sum_ci = sum([lnll_lst[i] for i in range(len(lnll_lst)) if labels[i] == c]) # s_bar(Ci | x) = sum(LN-LL) in cluster Ci
        clustered.append((int(c), sum_ci / Counter(labels)[c])) # s_bar(Ci | x) = sum(LN-LL) / count(Ci)
    total_mlsc = sum([x[1] for x in clustered])
    mlsc = [(c[0], c[1] / total_mlsc) for c in clustered]
    
    opt_cluster, opt_conf = max(mlsc, key=lambda x: x[1])
    optimal_response = max([(response_list[i], lnll_lst[i]) for i, label in enumerate(labels) if label == opt_cluster], key=lambda x: x[1])[0]

    return optimal_response, opt_conf


def bayesian_semantic_confidence(lnll_lst, response_list, labels):
    clustered = []
    for c in np.unique(labels):
        pi = Counter(labels)[c] / len(labels) # pi = count(Ci) / count
        joint_lnll = np.prod([lnll_lst[i] for i in range(len(lnll_lst)) if labels[i] == c]) # joint lnll = ∏(LN-LL) in cluster Ci
        clustered.append((int(c), joint_lnll * pi)) # joint_lnll * pi
    
    total_bsc = sum([x[1] for x in clustered])
    bsc = [(c[0], float(c[1] / total_bsc)) for c in clustered]

    opt_cluster, opt_conf = max(bsc, key=lambda x: x[1])
    optimal_response = max([(response_list[i], lnll_lst[i]) for i, label in enumerate(labels) if label == opt_cluster], key=lambda x: x[1])[0]

    return optimal_response, opt_conf