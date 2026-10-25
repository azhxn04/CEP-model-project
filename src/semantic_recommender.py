"""
Semantic Recommender Module
Computes dense semantic embeddings using SentenceTransformer ("all-MiniLM-L6-v2"),
calculates cosine similarity, requirement fit, budget fit, and generates explainable recommendations.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    # Resilient pure numpy fallback for environments without scikit-learn
    def cosine_similarity(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        a_norm = np.where(a_norm == 0, 1e-10, a_norm)
        b_norm = np.where(b_norm == 0, 1e-10, b_norm)
        return np.dot(a / a_norm, (b / b_norm).T)

    TfidfVectorizer = None

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "product_embeddings.npy")


class SemanticRecommender:
    def __init__(self, model_name: str = MODEL_NAME, fallback_tfidf: bool = True):
        self.model_name = model_name
        self.fallback_tfidf = fallback_tfidf
        self.model = None
        self.is_transformer_active = False
        self.tfidf_vectorizer = None
        self.catalog_df: Optional[pd.DataFrame] = None
        self.product_embeddings: Optional[np.ndarray] = None
        self._init_encoder()

    def _init_encoder(self):
        """Initializes the SentenceTransformer or sets up TF-IDF fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.is_transformer_active = True
            print(f"[SemanticRecommender] Successfully loaded SentenceTransformer ('{self.model_name}')")
        except Exception as e:
            print(f"[SemanticRecommender] SentenceTransformer not available or downloading ({e}). Using TF-IDF fallback.")
            self.is_transformer_active = False
            self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")

    def fit_catalog(self, catalog_df: pd.DataFrame, use_cache: bool = True):
        """Precomputes and caches embeddings for the entire product catalog."""
        self.catalog_df = catalog_df.copy().reset_index(drop=True)
        docs = self.catalog_df["Product Document"].tolist()

        if use_cache and os.path.exists(EMBEDDINGS_CACHE_PATH) and self.is_transformer_active:
            try:
                cached_embs = np.load(EMBEDDINGS_CACHE_PATH)
                if cached_embs.shape[0] == len(docs):
                    self.product_embeddings = cached_embs
                    print(f"[SemanticRecommender] Loaded {len(docs)} embeddings from cache.")
                    return
            except Exception as e:
                print(f"[SemanticRecommender] Cache load warning: {e}. Recomputing...")

        if self.is_transformer_active and self.model is not None:
            embeddings = self.model.encode(docs, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            self.product_embeddings = embeddings
            try:
                os.makedirs(os.path.dirname(EMBEDDINGS_CACHE_PATH), exist_ok=True)
                np.save(EMBEDDINGS_CACHE_PATH, self.product_embeddings)
                print(f"[SemanticRecommender] Precomputed & saved {len(docs)} embeddings to cache.")
            except Exception as e:
                print(f"[SemanticRecommender] Warning saving cache: {e}")
        else:
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self.product_embeddings = self.tfidf_vectorizer.fit_transform(docs).toarray()
            # Normalize
            norms = np.linalg.norm(self.product_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.product_embeddings = self.product_embeddings / norms

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query into a normalized dense vector."""
        if self.is_transformer_active and self.model is not None:
            emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            return emb[0]
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("Recommender has not been fit with catalog data.")
            emb = self.tfidf_vectorizer.transform([query]).toarray()[0]
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb

    def compute_similarity(self, query: str) -> np.ndarray:
        """Computes cosine similarity between customer query and all catalog products."""
        if self.product_embeddings is None or self.catalog_df is None:
            raise ValueError("Catalog embeddings not initialized. Call fit_catalog first.")

        query_vec = self.encode_query(query).reshape(1, -1)
        sim_scores = cosine_similarity(query_vec, self.product_embeddings)[0]
        # Clip scores to [0.0, 1.0]
        return np.clip(sim_scores, 0.0, 1.0)

    @staticmethod
    def calculate_budget_fit(price: float, budget: Optional[float]) -> float:
        """
        Calculates budget fit score:
        - 1.0 if within budget
        - Decays with penalty if over budget
        """
        if budget is None or budget <= 0:
            return 1.0
        if price <= budget:
            # Reward staying well within budget without being extremely underpowered
            return 1.0
        else:
            # Smooth decay: 10% over budget = 0.70, 20% over = 0.40, >50% over = 0.0
            ratio = (price - budget) / budget
            fit = max(0.0, 1.0 - (ratio * 3.0))
            return fit

    @staticmethod
    def calculate_requirement_fit(product_doc: str, suitable_for: str, keywords: List[str]) -> float:
        """Computes matching score between extracted query keywords and product features."""
        if not keywords:
            return 1.0
        text = (product_doc + " " + suitable_for).lower()
        matched = sum(1 for kw in keywords if kw.lower() in text)
        return matched / len(keywords)

    @staticmethod
    def generate_explanation(
        product: pd.Series,
        parsed_query: Dict[str, Any],
        semantic_score: float,
        budget_fit: float,
        req_fit: float,
        persona_note: Optional[str] = None
    ) -> str:
        """Generates clear, natural language explanation for why the product was recommended."""
        reasons = []
        if persona_note:
            reasons.append(persona_note)

        user_budget = parsed_query.get("budget")
        price = product.get("Selling Price (INR)", 0)
        keywords = parsed_query.get("keywords", [])
        suitable = product.get("Suitable For", "")

        # Category and Requirement alignment
        matched_specs = [kw for kw in keywords if kw.lower() in (str(product.get("Product Specs", "")) + " " + suitable).lower()]
        if matched_specs:
            reasons.append(f"Tailored for {', '.join(matched_specs[:3])}")
        elif suitable:
            reasons.append(f"Ideal for {suitable.split(',')[0].strip()}")

        # Budget fit
        if user_budget is not None and user_budget > 0:
            if price <= user_budget:
                savings = user_budget - price
                if savings > 0:
                    reasons.append(f"Fits within your ₹{user_budget:,.0f} budget (₹{savings:,.0f} under)")
                else:
                    reasons.append(f"Exactly matches your ₹{user_budget:,.0f} budget")
            else:
                reasons.append("Slightly above target budget but offers premium specifications")

        # Discount benefit
        discount = product.get("Discount Rate (%)", 0)
        if discount >= 15.0:
            reasons.append(f"Great value with {discount:.1f}% discount")

        # Semantic match confidence
        reasons.append(f"Semantic match confidence: {semantic_score * 100:.0f}%")

        return " • ".join(reasons)
