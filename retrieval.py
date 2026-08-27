import numpy as np


def eval_mAP(sorted_indices, query_labels, pool_labels):
    average_precisions = []
    for query_index in range(len(query_labels)):
        ranked_labels = pool_labels[sorted_indices[query_index]]
        relevant = ranked_labels == query_labels[query_index]
        hits = 0
        precisions = []
        for rank, is_relevant in enumerate(relevant, start=1):
            if is_relevant:
                hits += 1
                precisions.append(hits / rank)
        average_precisions.append(float(np.mean(precisions)) if precisions else 0.0)
    value = float(np.mean(average_precisions)) if average_precisions else 0.0
    return value


def eval_precision_recall(sorted_indices, query_labels, pool_labels, k=5):
    precisions = []
    for query_index in range(len(query_labels)):
        retrieved = pool_labels[sorted_indices[query_index]][:k]
        precisions.append(np.sum(retrieved == query_labels[query_index]) / k)
    avg_precision = float(np.mean(precisions)) if precisions else 0.0
    avg_recall = 0.0
    return avg_precision, avg_recall
