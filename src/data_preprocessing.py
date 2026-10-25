"""
Data Preprocessing Module
Handles loading, cleaning, and preparing product catalog, customer history,
and synthetic customer-product interaction logs for evaluation.
"""

import os
import pandas as pd
import numpy as np

DEFAULT_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "product_catalog.csv")
DEFAULT_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "customer_history.csv")
DEFAULT_INTERACTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "customer_interactions.csv")


def load_product_catalog(filepath: str = DEFAULT_CATALOG_PATH) -> pd.DataFrame:
    """Loads and standardizes the product catalog."""
    if not os.path.exists(filepath):
        # Fallback to current folder
        alt = os.path.join(os.path.dirname(__file__), "..", "product_catalog.csv")
        if os.path.exists(alt):
            filepath = alt
        else:
            raise FileNotFoundError(f"Catalog file not found at {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Ensure numeric columns
    numeric_cols = ["MRP (INR)", "Selling Price (INR)", "Discount Rate (%)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.replace("%", "").str.strip(), errors="coerce")

    # Fill NA
    df["Category"] = df["Category"].fillna("General")
    df["Item"] = df["Item"].fillna(df["Category"])
    df["Product Specs"] = df["Product Specs"].fillna("")
    df["Suitable For"] = df["Suitable For"].fillna("")

    # Construct clean Product Document if missing or incomplete
    if "Product Document" not in df.columns or df["Product Document"].isnull().any():
        df["Product Document"] = df.apply(
            lambda r: f"Product: {r['Item']}. Category: {r['Category']}. Specifications: {r['Product Specs']}. Suitable for: {r['Suitable For']}. Price: ₹{r['Selling Price (INR)']:,.0f}. Discount: {r['Discount Rate (%)']:.2f}%.",
            axis=1
        )

    return df


def load_customer_history(filepath: str = DEFAULT_HISTORY_PATH) -> pd.DataFrame:
    """Loads and cleans historical customer purchase data."""
    if not os.path.exists(filepath):
        alt = os.path.join(os.path.dirname(__file__), "..", "vijay_sales_customers_ML_enriched_2024_2025.csv")
        if os.path.exists(alt):
            filepath = alt
        else:
            raise FileNotFoundError(f"Customer history file not found at {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Clean numeric fields
    numeric_cols = ["Age", "Purchase Amount (INR)", "Quantity", "Time Spent (Minutes)", "MRP (INR)", "Selling Price (INR)", "Discount Rate (%)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.replace("%", "").str.strip(), errors="coerce")

    # Date parsing
    if "Purchase Date" in df.columns:
        df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors="coerce")
        df["Year"] = df["Purchase Date"].dt.year
        df["Month"] = df["Purchase Date"].dt.month
        df["DayOfWeek"] = df["Purchase Date"].dt.day_name()
        df["YearMonth"] = df["Purchase Date"].dt.to_period("M").astype(str)

    return df


def generate_synthetic_interactions(catalog_df: pd.DataFrame, num_interactions: int = 600, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic held-out interaction dataset for pure recommendation metrics
    (Precision@K, Recall@K, Hit Rate@K, NDCG@K) without label leakage.
    Simulates queries, retrieved candidates, impressions, clicks, and purchases.
    """
    np.random.seed(random_state)

    query_templates = [
        ("I need a laptop for Python and data analysis under 60k", "Laptop", 60000, ["Coding", "Python", "data analysis"]),
        ("Looking for a gaming laptop with dedicated GPU under 80000", "Laptop", 80000, ["gaming", "dedicated graphics"]),
        ("Need a budget laptop for college and office under 50000", "Laptop", 50000, ["college", "office work"]),
        ("Front load washing machine for large family under 45000", "Washing Machine", 45000, ["Large families", "front load"]),
        ("Affordable top load washing machine below 25k", "Washing Machine", 25000, ["top load", "Small families"]),
        ("Double door frost free refrigerator for family under 35000", "Refrigerator", 35000, ["frost free", "double door"]),
        ("Single door compact fridge for small room under 20k", "Refrigerator", 20000, ["single door", "students"]),
        ("5G mobile phone with high resolution 50MP camera under 30000", "Mobile Phone", 30000, ["5G", "camera"]),
        ("High performance gaming phone with 12GB RAM under 40000", "Mobile Phone", 40000, ["Gaming", "multitasking"]),
        ("Study tablet with stylus support for taking notes under 35000", "Tablet", 35000, ["stylus", "notes"]),
        ("Budget tablet for online classes and reading under 20000", "Tablet", 20000, ["online classes", "basic study"]),
        ("4K Smart TV 55 inch with Dolby Audio for home theater under 50000", "Television", 50000, ["55 inch", "Dolby Audio"]),
        ("Compact 32 inch HD TV for bedroom under 20000", "Television", 20000, ["32 inch", "Bedroom"]),
        ("1.5 Ton 5 Star split inverter AC for master bedroom under 50000", "Air Conditioner", 50000, ["1.5 Ton", "5 Star"]),
        ("1 Ton energy efficient AC for small room under 35k", "Air Conditioner", 35000, ["1 Ton", "Small rooms"]),
        ("Energy saving BLDC ceiling fan with remote control under 4000", "Fan", 4000, ["BLDC", "remote"]),
        ("PS5 Slim console for 4K gaming under 50000", "Gaming Console", 50000, ["PS5", "4K gaming"]),
        ("Loud party speaker with deep bass and bluetooth under 15000", "Speaker", 15000, ["party", "deep bass"]),
        ("Convection microwave oven with grill for baking under 15000", "Oven", 15000, ["convection", "Baking"]),
        ("Cordless stick vacuum cleaner with HEPA filter for pet hair under 15000", "Vacuum Cleaner", 15000, ["Cordless", "pets"]),
        ("RO UV water purifier with mineralizer for hard water under 15000", "Water Purifier", 15000, ["RO+UV", "hard water"]),
        ("65W GaN fast charger with USB-C PD under 3500", "Charger", 3500, ["65W GaN", "fast charging"])
    ]

    records = []
    customer_ids = [f"USR_{1000 + i}" for i in range(150)]

    for i in range(num_interactions):
        q_idx = np.random.randint(len(query_templates))
        query_text, category, budget, keywords = query_templates[q_idx]
        user_id = np.random.choice(customer_ids)

        # Match relevant products in catalog
        matching_prods = catalog_df[
            (catalog_df["Category"].str.lower() == category.lower()) | 
            (catalog_df["Item"].str.lower() == category.lower())
        ]

        if matching_prods.empty:
            matching_prods = catalog_df

        # Score candidates for ground truth simulation
        candidate_scores = []
        for _, prod in matching_prods.iterrows():
            score = 0.0
            price = prod["Selling Price (INR)"]
            if price <= budget:
                score += 0.4
            else:
                score -= min(0.3, (price - budget) / budget)
            
            doc_lower = prod["Product Document"].lower()
            for kw in keywords:
                if kw.lower() in doc_lower:
                    score += 0.2
            candidate_scores.append((prod["Product ID"], max(0.01, score)))

        # Pick ground truth relevant product ID
        pids = [c[0] for c in candidate_scores]
        probs = np.array([c[1] for c in candidate_scores])
        probs = probs / probs.sum()
        ground_truth_pid = np.random.choice(pids, p=probs)

        # Simulate user behavior
        clicked = 1 if np.random.rand() < 0.82 else 0
        purchased = 1 if (clicked == 1 and np.random.rand() < 0.65) else 0
        rating = np.random.choice([4, 5], p=[0.35, 0.65]) if purchased == 1 else (np.random.choice([2, 3, 4]) if clicked == 1 else np.nan)

        records.append({
            "Interaction_ID": f"INT_{i+1:04d}",
            "Customer_ID": user_id,
            "Query": query_text,
            "Target_Category": category,
            "Budget": budget,
            "Ground_Truth_Product_ID": ground_truth_pid,
            "Clicked": clicked,
            "Purchased": purchased,
            "Rating": rating,
            "Timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(days=np.random.randint(0, 365), hours=np.random.randint(8, 22))
        })

    interactions_df = pd.DataFrame(records)
    return interactions_df
