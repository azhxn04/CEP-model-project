"""
Generate the 3 Jupyter Notebooks for the project:
1. 01_data_analysis.ipynb
2. 02_nlp_recommender.ipynb
3. 03_ml_model_evaluation.ipynb
"""

import json
import os

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def code_cell(source):
    if isinstance(source, list):
        src = [s + "\n" if not s.endswith("\n") else s for s in source]
    else:
        src = [s + "\n" for s in source.split("\n")]
    # remove trailing newline on last line
    if src:
        src[-1] = src[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src
    }

def md_cell(source):
    if isinstance(source, list):
        src = [s + "\n" if not s.endswith("\n") else s for s in source]
    else:
        src = [s + "\n" for s in source.split("\n")]
    if src:
        src[-1] = src[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src
    }

def generate_notebooks():
    nb_dir = os.path.join(os.path.dirname(__file__), "notebooks")
    os.makedirs(nb_dir, exist_ok=True)

    # -------------------------------------------------------------
    # NOTEBOOK 1: 01_data_analysis.ipynb
    # -------------------------------------------------------------
    nb1_cells = [
        md_cell("""# 01 — Exploratory Data Analysis (EDA) & Dataset Validation
## AI-Powered Natural-Language Product Recommendation System

This notebook performs end-to-end exploratory analysis and data quality validation across:
1. **Product Catalog** (40 products across electronics categories)
2. **Customer Historical Transactions** (500 synthetic purchase records covering 2024–2025)

---
### Objectives:
* Inspect schemas, data types, null values, and summary statistics.
* Analyze category distributions, pricing dynamics, and discount strategies.
* Explore customer demographics (Age, Gender) and shopping behaviors (Channel, Time Spent, Shift).
* Identify key trends and prepare clean data for downstream NLP and ML components.
"""),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.abspath('..'))
from src.data_preprocessing import load_product_catalog, load_customer_history

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11"""),
        md_cell("""## 1. Product Catalog Analysis"""),
        code_cell("""catalog_df = load_product_catalog('../data/product_catalog.csv')
print(f"Product Catalog Shape: {catalog_df.shape}")
catalog_df.head()"""),
        code_cell("""catalog_df.info()
print("\\nMissing values per column:\\n", catalog_df.isnull().sum())"""),
        code_cell("""catalog_df[['MRP (INR)', 'Selling Price (INR)', 'Discount Rate (%)']].describe()"""),
        code_cell("""# Product distribution across categories
plt.figure(figsize=(10, 5))
cat_counts = catalog_df['Category'].value_counts()
sns.barplot(x=cat_counts.values, y=cat_counts.index, palette='Blues_r')
plt.title('Product Count by Category in Catalog', fontsize=14, fontweight='bold')
plt.xlabel('Number of Products')
plt.ylabel('Category')
plt.show()"""),
        code_cell("""# Average price and discount by category
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

avg_price = catalog_df.groupby('Category')['Selling Price (INR)'].mean().sort_values(ascending=False)
sns.barplot(ax=axes[0], x=avg_price.values, y=avg_price.index, palette='crest')
axes[0].set_title('Average Selling Price by Category (INR)', fontweight='bold')
axes[0].set_xlabel('Price (₹)')

avg_disc = catalog_df.groupby('Category')['Discount Rate (%)'].mean().sort_values(ascending=False)
sns.barplot(ax=axes[1], x=avg_disc.values, y=avg_disc.index, palette='flare')
axes[1].set_title('Average Discount Rate (%) by Category', fontweight='bold')
axes[1].set_xlabel('Discount (%)')

plt.tight_layout()
plt.show()"""),
        md_cell("""## 2. Historical Customer Transactions Analysis"""),
        code_cell("""history_df = load_customer_history('../data/customer_history.csv')
print(f"Customer History Shape: {history_df.shape}")
history_df.head()"""),
        code_cell("""history_df.info()
print("\\nMissing values in customer history:\\n", history_df.isnull().sum())"""),
        code_cell("""history_df[['Age', 'Purchase Amount (INR)', 'Quantity', 'Time Spent (Minutes)', 'Discount Rate (%)']].describe()"""),
        code_cell("""# Customer Demographics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Age distribution
sns.histplot(ax=axes[0], data=history_df, x='Age', bins=15, kde=True, color='#3B82F6')
axes[0].set_title('Customer Age Distribution', fontweight='bold')

# Gender distribution
history_df['Gender'].value_counts().plot.pie(ax=axes[1], autopct='%1.1f%%', colors=['#60A5FA', '#F472B6'])
axes[1].set_title('Customer Gender Share', fontweight='bold')
axes[1].set_ylabel('')

# Purchase Mode
history_df['Purchase Mode'].value_counts().plot.pie(ax=axes[2], autopct='%1.1f%%', colors=['#34D399', '#FBBF24'])
axes[2].set_title('Online vs. Offline Store Share', fontweight='bold')
axes[2].set_ylabel('')

plt.tight_layout()
plt.show()"""),
        code_cell("""# Purchases over time trend
if 'YearMonth' in history_df.columns:
    monthly = history_df.groupby('YearMonth').size()
    plt.figure(figsize=(12, 5))
    monthly.plot(marker='o', color='#2563EB', linewidth=2.5)
    plt.title('Monthly Historical Purchase Volume (2024–2025)', fontsize=14, fontweight='bold')
    plt.xlabel('Year-Month')
    plt.ylabel('Transactions')
    plt.xticks(rotation=45)
    plt.show()"""),
        code_cell("""# Category popularity in historical data
plt.figure(figsize=(10, 6))
item_counts = history_df['Item'].value_counts()
sns.barplot(x=item_counts.values, y=item_counts.index, palette='viridis')
plt.title('Total Purchases by Item Category', fontsize=14, fontweight='bold')
plt.xlabel('Count of Purchases')
plt.show()"""),
        md_cell("""## 3. Key Findings & Insights Summary
1. **Catalog Diversity**: 40 products evenly span 14 key categories (Laptops, Appliances, TVs, Phones, Consoles, etc.).
2. **Pricing Structure**: Wide pricing spectrum from ₹1,399 (Chargers) to ₹77,999 (i7 Laptops).
3. **Discount Strategy**: Aggressive discounting on fans and ovens (~25-30%), with moderate discounts on premium electronics (~9-15%).
4. **Customer Demographic**: Ages range from 18 to 65 with balanced gender and omnichannel (50/50 Online vs. Store) split.
""")
    ]

    with open(os.path.join(nb_dir, "01_data_analysis.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb1_cells), f, indent=2)

    # -------------------------------------------------------------
    # NOTEBOOK 2: 02_nlp_recommender.ipynb
    # -------------------------------------------------------------
    nb2_cells = [
        md_cell("""# 02 — NLP & Semantic Recommendation Engine
## AI-Powered Natural-Language Product Recommendation System

This notebook demonstrates the core Natural Language Processing and Semantic Search pipeline:
1. **Query Understanding & Parsing**: Extracting product category, budget constraints, and specification keywords.
2. **Dense Vector Embeddings**: Using Pretrained Sentence Transformers (`all-MiniLM-L6-v2`) and vector representations.
3. **Cosine Similarity Computation**: Comparing customer query embeddings against Product Documents.
4. **Multi-Signal Hybrid Ranking**: Combining semantic similarity, budget fit, requirement fit, and discounts.
5. **Explainable AI**: Generating transparent natural-language reasons for every recommendation.
"""),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('..'))
from src.data_preprocessing import load_product_catalog
from src.query_parser import parse_query, extract_budget, detect_category, extract_keywords
from src.semantic_recommender import SemanticRecommender
from src.ranking import HybridRecommender

catalog_df = load_product_catalog('../data/product_catalog.csv')
print(f"Catalog loaded with {len(catalog_df)} products.")"""),
        md_cell("""## 1. Natural Language Query Parsing & Entity Extraction"""),
        code_cell("""test_queries = [
    "I need a laptop for Python and data analysis under ₹60,000",
    "Suggest a washing machine for a family under 40k",
    "I want a phone with a good camera under 30000",
    "1.5 Ton 5 Star split inverter AC for bedroom below 50k",
    "PS5 console for 4K gaming under ₹50,000",
    "Budget tablet for online study and notes under 20000"
]

print("=== Query Understanding Results ===\\n")
for q in test_queries:
    parsed = parse_query(q)
    print(f"Query: '{q}'")
    print(f"  -> Category Detected : {parsed['category']}")
    print(f"  -> Budget Extracted  : ₹{parsed['budget']:,.0f}" if parsed['budget'] else "  -> Budget: None")
    print(f"  -> Key Specs/Features: {parsed['keywords']}")
    print("-" * 60)"""),
        md_cell("""## 2. Semantic Embedding & Recommender Engine Initialization"""),
        code_cell("""semantic_engine = SemanticRecommender()
semantic_engine.fit_catalog(catalog_df)

hybrid_recommender = HybridRecommender(
    catalog_df=catalog_df,
    semantic_engine=semantic_engine
)
print("Hybrid recommender initialized successfully.")"""),
        md_cell("""## 3. End-to-End Recommendation & Ranking Demonstration"""),
        code_cell("""sample_query = "I want a laptop with 16GB RAM for coding and data science under ₹60,000"
rec_result = hybrid_recommender.recommend(
    query=sample_query,
    top_k=3,
    strict_category=True
)

print(f"Query: {rec_result['query']}")
print(f"Total Candidate Pool: {rec_result['total_candidates']}")
print("\\n--- Top 3 Recommendations ---")
for i, item in enumerate(rec_result['recommendations'], 1):
    print(f"\\n#{i}: {item['item']} ({item['product_id']})")
    print(f"   Specs        : {item['specs']}")
    print(f"   Selling Price: ₹{item['selling_price']:,.0f} (MRP: ₹{item['mrp']:,.0f}, Save {item['discount_rate']:.1f}%)")
    print(f"   Score Matrix : Hybrid={item['hybrid_score']:.3f} | Semantic={item['semantic_score']:.3f} | BudgetFit={item['budget_fit']:.2f} | ReqFit={item['requirement_fit']:.2f}")
    print(f"   Explanation  : {item['explanation']}")"""),
        code_cell("""# Testing across multiple customer intents
test_intents = [
    "Washing machine for a large family with steam wash under 45k",
    "Compact single door refrigerator for student room under 20000",
    "Portable bluetooth party speaker with deep bass"
]

for q in test_intents:
    res = hybrid_recommender.recommend(q, top_k=2)
    print(f"\\n================ QUERY: '{q}' ================")
    for rank, p in enumerate(res['recommendations'], 1):
        print(f"  #{rank} [{p['category']}] {p['item']} | ₹{p['selling_price']:,.0f} | Match: {p['hybrid_score']*100:.1f}%")
        print(f"     Why: {p['explanation']}")""")
    ]

    with open(os.path.join(nb_dir, "02_nlp_recommender.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb2_cells), f, indent=2)

    # -------------------------------------------------------------
    # NOTEBOOK 3: 03_ml_model_evaluation.ipynb
    # -------------------------------------------------------------
    nb3_cells = [
        md_cell("""# 03 — Machine Learning & Dynamic Recommendation Evaluation
## AI-Powered Natural-Language Product Recommendation System

This notebook covers the machine learning modeling, label leakage prevention, and technical evaluation:
1. **Critical ML Problem — Label Leakage Prevention**: Why predicting target items from synthetic requirement text produces invalid 100% artificial shortcuts.
2. **Behavioral Classification Models**:
   * Baseline 1: Logistic Regression
   * Baseline 2: Random Forest Classifier
   * Primary Model: XGBoost Classifier
3. **Classification Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC on an 80/20 Stratified Test Set.
4. **Pure Recommendation Evaluation**: Precision@K, Recall@K, Hit Rate@K, NDCG@K, and MRR@K on held-out user query interactions.
"""),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath('..'))
from src.data_preprocessing import load_customer_history, load_product_catalog, generate_synthetic_interactions
from src.ml_model import CustomerMLModel
from src.semantic_recommender import SemanticRecommender
from src.ranking import HybridRecommender
from src.evaluation import evaluate_recommender_on_interactions

sns.set_theme(style="whitegrid")"""),
        md_cell("""## 1. Machine Learning Training on Independent Behavioral Features"""),
        code_cell("""history_df = load_customer_history('../data/customer_history.csv')
ml_system = CustomerMLModel()
ml_results = ml_system.train_and_evaluate(history_df, test_size=0.20, random_state=42)

# Display tabular model metrics
metrics_df = pd.DataFrame([
    {
        "Model": name,
        "Accuracy (%)": f"{vals['accuracy']*100:.2f}%",
        "F1 Score (%)": f"{vals['f1_score']*100:.2f}%",
        "Precision (%)": f"{vals['precision']*100:.2f}%",
        "Recall (%)": f"{vals['recall']*100:.2f}%",
        "Multiclass ROC-AUC": f"{vals.get('roc_auc', 0.5):.4f}" if vals.get('roc_auc') else "N/A"
    }
    for name, vals in ml_results["metrics"].items()
])
metrics_df"""),
        md_cell("""## 2. Confusion Matrix Heatmap (XGBoost)"""),
        code_cell("""cm = ml_results["metrics"]["XGBoost"]["confusion_matrix"]
classes = ml_results["classes"]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Multi-Class Confusion Matrix — XGBoost Classifier', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Product Class')
plt.ylabel('Actual Product Class')
plt.xticks(rotation=45)
plt.show()"""),
        md_cell("""## 3. Feature Importance (XGBoost)"""),
        code_cell("""feat_imp = ml_results["metrics"]["XGBoost"].get("feature_importance", {})
if feat_imp:
    feat_series = pd.Series(feat_imp).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    feat_series.plot(kind='barh', color='#3B82F6')
    plt.title('XGBoost Feature Importance (Demographic & Behavioral Signals)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.show()"""),
        md_cell("""## 4. Pure Recommendation Ranking Metrics Evaluation (Top 1, Top 3, Top 5)"""),
        code_cell("""catalog_df = load_product_catalog('../data/product_catalog.csv')
semantic_engine = SemanticRecommender()
semantic_engine.fit_catalog(catalog_df)

hybrid_engine = HybridRecommender(
    catalog_df=catalog_df,
    semantic_engine=semantic_engine,
    ml_model=ml_system
)

# Generate or load held-out interaction dataset
interactions_path = '../data/customer_interactions.csv'
if os.path.exists(interactions_path):
    interactions_df = pd.read_csv(interactions_path)
else:
    interactions_df = generate_synthetic_interactions(catalog_df, num_interactions=500)

rec_eval = evaluate_recommender_on_interactions(hybrid_engine, interactions_df, k_values=[1, 3, 5])

print("=== Recommendation Ranking Metrics on Held-Out Test Set ===")
summary = rec_eval["summary"]
table_data = [
    {"Recommendation Depth": "Top 1", "Precision": f"{summary['Precision@1']*100:.2f}%", "Recall": f"{summary['Recall@1']*100:.2f}%", "Hit Rate": f"{summary['Hit_Rate@1']*100:.2f}%", "NDCG Score": f"{summary['NDCG@1']:.4f}"},
    {"Recommendation Depth": "Top 3", "Precision": f"{summary['Precision@3']*100:.2f}%", "Recall": f"{summary['Recall@3']*100:.2f}%", "Hit Rate": f"{summary['Hit_Rate@3']*100:.2f}%", "NDCG Score": f"{summary['NDCG@3']:.4f}"},
    {"Recommendation Depth": "Top 5", "Precision": f"{summary['Precision@5']*100:.2f}%", "Recall": f"{summary['Recall@5']*100:.2f}%", "Hit Rate": f"{summary['Hit_Rate@5']*100:.2f}%", "NDCG Score": f"{summary['NDCG@5']:.4f}"}
]
pd.DataFrame(table_data)"""),
        code_cell("""# Plotting Recommendation Metrics Growth across Top-N Results
summary = rec_eval["summary"]
plot_rec_df = pd.DataFrame({
    "Results Depth": ["Top 1", "Top 3", "Top 5"],
    "Recall": [summary["Recall@1"], summary["Recall@3"], summary["Recall@5"]],
    "Hit Rate": [summary["Hit_Rate@1"], summary["Hit_Rate@3"], summary["Hit_Rate@5"]],
    "NDCG Score": [summary["NDCG@1"], summary["NDCG@3"], summary["NDCG@5"]]
})

plot_rec_df.set_index("Results Depth").plot(kind='bar', figsize=(10, 5), color=['#3B82F6', '#10B981', '#8B5CF6'])
plt.title('Recommendation Metrics across Top-N Results', fontsize=14, fontweight='bold')
plt.ylabel('Score (0.0 to 1.0)')
plt.ylim(0, 1.15)
plt.xticks(rotation=0)
plt.legend(title='Metric')
plt.show()""")
    ]

    with open(os.path.join(nb_dir, "03_ml_model_evaluation.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb3_cells), f, indent=2)

    print("Successfully generated all 3 Jupyter notebooks in 'notebooks/'")

if __name__ == "__main__":
    generate_notebooks()
