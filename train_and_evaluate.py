"""
Offline Training, Embeddings Precomputation, and Model Evaluation Pipeline
Executes the full machine learning training & evaluation workflow, saving
all model artifacts, performance metrics, and precomputed embeddings for the Streamlit UI.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from src.data_preprocessing import (
    load_product_catalog,
    load_customer_history,
    generate_synthetic_interactions
)
from src.ml_model import CustomerMLModel
from src.semantic_recommender import SemanticRecommender
from src.ranking import HybridRecommender
from src.evaluation import evaluate_recommender_on_interactions

def run_pipeline():
    print("=" * 70)
    print("STARTING AI PRODUCT RECOMMENDER TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1/5] Loading and cleaning datasets...")
    catalog_df = load_product_catalog()
    history_df = load_customer_history()
    print(f"  • Product Catalog: {len(catalog_df)} products loaded.")
    print(f"  • Customer History: {len(history_df)} historical transactions loaded.")

    # 2. Train Tabular ML Models (Logistic Regression, Random Forest, XGBoost)
    print("\n[Step 2/5] Training structured ML models (Preventing Label Leakage)...")
    ml_system = CustomerMLModel()
    ml_results = ml_system.train_and_evaluate(history_df, test_size=0.20, random_state=42)
    
    print("\n--- ML Model Performance on Held-Out Test Set (20% Stratified) ---")
    for model_name, metrics in ml_results["metrics"].items():
        print(f"  • {model_name:20s} | Accuracy: {metrics['accuracy']*100:.2f}% | F1-Score: {metrics['f1_score']*100:.2f}% | Precision: {metrics['precision']*100:.2f}% | Recall: {metrics['recall']*100:.2f}%")

    # 3. Generate Held-Out Interactions Dataset for Pure Recommendation Metrics
    print("\n[Step 3/5] Generating held-out customer query interactions for recommendation evaluation...")
    interactions_df = generate_synthetic_interactions(catalog_df, num_interactions=500, random_state=42)
    interactions_path = os.path.join(os.path.dirname(__file__), "data", "customer_interactions.csv")
    interactions_df.to_csv(interactions_path, index=False)
    print(f"  • Saved {len(interactions_df)} ground-truth test interactions to {interactions_path}")

    # 4. Initialize Semantic Recommender & Hybrid Ranking Engine
    print("\n[Step 4/5] Precomputing semantic catalog embeddings...")
    semantic_engine = SemanticRecommender()
    semantic_engine.fit_catalog(catalog_df)
    
    hybrid_engine = HybridRecommender(
        catalog_df=catalog_df,
        semantic_engine=semantic_engine,
        ml_model=ml_system
    )

    # 5. Evaluate Recommendation Ranking Metrics
    print("\n[Step 5/5] Evaluating pure recommendation ranking metrics (Precision@K, Recall@K, NDCG@K, MRR)...")
    rec_eval = evaluate_recommender_on_interactions(hybrid_engine, interactions_df, k_values=[1, 3, 5])
    
    print("\n--- Recommendation Ranking Performance ---")
    for metric, score in rec_eval["summary"].items():
        print(f"  • {metric:16s}: {score:.4f} ({score*100:.2f}%)")

    # Save comprehensive evaluation package for Streamlit UI
    full_eval_artifact = {
        "ml_evaluation": ml_results,
        "recommendation_evaluation": rec_eval,
        "catalog_summary": {
            "num_products": len(catalog_df),
            "categories": sorted(list(catalog_df["Category"].unique())),
            "price_range": [float(catalog_df["Selling Price (INR)"].min()), float(catalog_df["Selling Price (INR)"].max())],
            "avg_discount": float(catalog_df["Discount Rate (%)"].mean())
        },
        "history_summary": {
            "num_records": len(history_df),
            "age_range": [int(history_df["Age"].min()), int(history_df["Age"].max())],
            "avg_time_spent": float(history_df["Time Spent (Minutes)"].mean())
        }
    }

    eval_artifact_path = os.path.join(os.path.dirname(__file__), "models", "complete_system_metrics.joblib")
    joblib.dump(full_eval_artifact, eval_artifact_path)
    print(f"\nSaved complete evaluation artifact to {eval_artifact_path}")
    print("=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
