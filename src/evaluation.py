"""
Evaluation Metrics Module
Distinguishes Classification Evaluation from Pure Recommendation Ranking Metrics.
Includes:
- Classification: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
- Recommendation: Precision@K, Recall@K, Hit Rate@K, NDCG@K, MRR@K
"""

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Set, Tuple


def precision_at_k(recommended_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """Calculates Precision@K = (Relevant items in Top-K) / K."""
    if k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    relevant_in_top_k = sum(1 for pid in top_k if pid in ground_truth_ids)
    return relevant_in_top_k / float(k)


def recall_at_k(recommended_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """Calculates Recall@K = (Relevant items in Top-K) / Total Relevant Items."""
    if not ground_truth_ids or k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    relevant_in_top_k = sum(1 for pid in top_k if pid in ground_truth_ids)
    return relevant_in_top_k / float(len(ground_truth_ids))


def hit_rate_at_k(recommended_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """Calculates Hit Rate@K = 1.0 if at least one relevant item is in Top-K, else 0.0."""
    top_k = recommended_ids[:k]
    for pid in top_k:
        if pid in ground_truth_ids:
            return 1.0
    return 0.0


def ndcg_at_k(recommended_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """Calculates Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance."""
    if not ground_truth_ids or k <= 0:
        return 0.0

    dcg = 0.0
    top_k = recommended_ids[:k]
    for i, pid in enumerate(top_k):
        rel = 1.0 if pid in ground_truth_ids else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 2)

    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(ground_truth_ids))))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr_at_k(recommended_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """Calculates Mean Reciprocal Rank (MRR@K)."""
    top_k = recommended_ids[:k]
    for i, pid in enumerate(top_k):
        if pid in ground_truth_ids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_recommender_on_interactions(
    recommender,
    interactions_df: pd.DataFrame,
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, Any]:
    """
    Evaluates recommendation system against ground-truth customer interactions.
    Computes Precision@K, Recall@K, Hit Rate@K, NDCG@K, MRR across test queries.
    """
    results = {k: {"precision": [], "recall": [], "hit_rate": [], "ndcg": [], "mrr": []} for k in k_values}
    per_query_records = []

    for _, row in interactions_df.iterrows():
        query = str(row["Query"])
        ground_truth_pid = str(row["Ground_Truth_Product_ID"])
        ground_truth_set = {ground_truth_pid}

        rec_out = recommender.recommend(query=query, top_k=max(k_values))
        recommended_pids = [item["product_id"] for item in rec_out["recommendations"]]

        query_record = {
            "query": query,
            "ground_truth": ground_truth_pid,
            "recommended": recommended_pids[:3]
        }

        for k in k_values:
            p_k = precision_at_k(recommended_pids, ground_truth_set, k=k)
            r_k = recall_at_k(recommended_pids, ground_truth_set, k=k)
            h_k = hit_rate_at_k(recommended_pids, ground_truth_set, k=k)
            n_k = ndcg_at_k(recommended_pids, ground_truth_set, k=k)
            m_k = mrr_at_k(recommended_pids, ground_truth_set, k=k)

            results[k]["precision"].append(p_k)
            results[k]["recall"].append(r_k)
            results[k]["hit_rate"].append(h_k)
            results[k]["ndcg"].append(n_k)
            results[k]["mrr"].append(m_k)

            query_record[f"hit@{k}"] = h_k
            query_record[f"ndcg@{k}"] = n_k

        per_query_records.append(query_record)

    summary = {}
    for k in k_values:
        summary[f"Precision@{k}"] = float(np.mean(results[k]["precision"]))
        summary[f"Recall@{k}"] = float(np.mean(results[k]["recall"]))
        summary[f"Hit_Rate@{k}"] = float(np.mean(results[k]["hit_rate"]))
        summary[f"NDCG@{k}"] = float(np.mean(results[k]["ndcg"]))
        summary[f"MRR@{k}"] = float(np.mean(results[k]["mrr"]))

    return {
        "summary": summary,
        "sample_queries": per_query_records[:10],
        "total_test_queries": len(interactions_df)
    }
