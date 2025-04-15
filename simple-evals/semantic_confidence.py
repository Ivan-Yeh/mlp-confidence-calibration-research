import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from .common import (
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    normalize_extracted_answer,
    normalize_response,
    ANSWER_PATTERN_MULTICHOICE
)
import re
# ---------------------------------------------------------
# model_name="all-MiniLM-L6-v2"
# distance_threshold = 0.7
# lnll_lst = [(x)[1] for x in response_sample]
# response_list = [x[0] for x in response_sample]
# embeddings = SentenceTransformer(model_name).encode(response_list)
# clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average")
# labels = clustering.fit_predict(embeddings)
# ---------------------------------------------------------

def mmlu_regex_extract_response(response_text):
    response_text = normalize_response(response_text)
    extracted_answer = None
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1))
            break
    return extracted_answer

def gpqa_regex_extract_response(response_text):
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    extracted_answer = match.group(1) if match else None
    return extracted_answer

mcq_regex_extractors = {"mmlu": mmlu_regex_extract_response, "gpqa": gpqa_regex_extract_response}

def get_semantic_clusters(multi_response):
    lnll_lst = [(x)[1] for x in multi_response]
    response_list = [x[0] for x in multi_response]
    distance_threshold = 0.3
    model_name="all-MiniLM-L6-v2"
    embeddings = SentenceTransformer(model_name).encode(response_list)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average")
    labels = clustering.fit_predict(embeddings)
    return response_list, lnll_lst, labels


def get_mcq_clusters(multi_response, test = "mmlu"):
    lnll_lst = [(x)[1] for x in multi_response]
    response_list = [x[0] for x in multi_response]
    choice_map = dict()
    choice_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    labels = [choice_map.get(mcq_regex_extractors[test](c), 0) for c in response_list]
    return response_list, lnll_lst, labels


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