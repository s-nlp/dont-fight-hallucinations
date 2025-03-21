import itertools

import numpy as np
from sklearn.cluster import KMeans


def get_ce_score(pairs, nli_model):
    scores = nli_model.predict(pairs)
    label_names = ['contradiction', 'entailment', 'neutral']
    predictions = []
    for pred in scores:
        predictions.append({name: round(float(pred), 2) for pred, name in zip(pred, label_names)})
    return predictions


def get_custom_scores(input, nli_model):
    pairs = list(itertools.combinations(input, r=2))
    
    formatted_pairs = [(pair[0], pair[1]) for pair in pairs]
    scores_1 = get_ce_score(formatted_pairs, nli_model)

    formatted_pairs = [(pair[1], pair[0]) for pair in pairs]
    scores_2 = get_ce_score(formatted_pairs, nli_model)
    return scores_1, scores_2


def weighted_agg(input, ent_w, cont_w, neutral_w):
    ent = input['entailment']
    cont = input['contradiction']
    neutral = input['neutral']

    weighted_sum = ent_w * ent + neutral_w * neutral + cont_w * cont
    return weighted_sum


def aggregation(scores_1, scores_2, ent_w=1, cont_w=-1, neutral_w=0):
    agg_1 = [weighted_agg(s, ent_w, cont_w, neutral_w) for s in scores_1]
    agg_2 = [weighted_agg(s, ent_w, cont_w, neutral_w) for s in scores_2]

    agg = np.array(agg_1 + agg_2)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(agg.reshape(-1, 1))
    centroids = kmeans.cluster_centers_
    return np.min(centroids)
