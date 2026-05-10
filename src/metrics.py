import numpy as np


def recall_metric(ground_truth, preds):
    num_gt = len(ground_truth)
    if num_gt == 0:
        return 0

    relevant = list(set(ground_truth) & set(preds))
    num_relevant = len(relevant)

    return num_relevant / num_gt


def dcg_calc(scores):
    num = np.power(2, scores) - 1
    denom = np.log2(np.arange(scores.shape[0], dtype=np.float64) + 2)
    dcg = np.sum(num / denom)

    return dcg


def ndcg_metric(ground_truth, preds):
    num_preds = len(preds)
    if num_preds == 0:
        return 0

    relevant = np.array([1 if x in ground_truth else 0 for x in preds])

    dcg = dcg_calc(relevant)
    if dcg == 0.0:
        return 0.0

    ideal_relevant = np.zeros(num_preds)
    num_relevant = min(num_preds, len(ground_truth))
    ideal_relevant[:num_relevant] = 1

    ideal_dcg = dcg_calc(ideal_relevant)

    if ideal_dcg == 0.0:
        return 0.0

    normalized_dcg = dcg / ideal_dcg

    return normalized_dcg


def evaluate_model(test_df, preds_column, gt_column, top_k=10):
    metrics = []

    for idx, row in test_df.iterrows():
        gt_items = row[gt_column]
        preds = row[preds_column][:top_k]

        ndcg = ndcg_metric(gt_items, preds)
        recall = recall_metric(gt_items, preds)

        metrics.append((ndcg, recall))

    mean_ndcg = np.mean([x[0] for x in metrics])
    mean_recall = np.mean([x[1] for x in metrics])

    return mean_ndcg, mean_recall
