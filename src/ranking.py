"""
Hybrid Recommendation Ranking Module
Combines Semantic Similarity, Budget Fit, Requirement Fit, Discount Signal, and Behavioral ML Score
into a multi-criteria hybrid ranking pipeline.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.query_parser import parse_query
from src.semantic_recommender import SemanticRecommender
from src.ml_model import CustomerMLModel


# Default tuned weights across signals
DEFAULT_WEIGHTS = {
    "semantic": 0.45,
    "budget_fit": 0.20,
    "requirement_fit": 0.15,
    "discount": 0.10,
    "behavioral_ml": 0.10
}


class HybridRecommender:
    def __init__(
        self,
        catalog_df: pd.DataFrame,
        semantic_engine: Optional[SemanticRecommender] = None,
        ml_model: Optional[CustomerMLModel] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.catalog_df = catalog_df.copy().reset_index(drop=True)
        self.weights = weights or DEFAULT_WEIGHTS.copy()

        # Normalize weights
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

        # Initialize or assign semantic recommender
        if semantic_engine is not None:
            self.semantic_engine = semantic_engine
        else:
            self.semantic_engine = SemanticRecommender()
            self.semantic_engine.fit_catalog(self.catalog_df)

        # Assign ML model
        self.ml_model = ml_model

    def compute_persona_affinity(
        self,
        customer_profile: Optional[Dict[str, Any]],
        ml_probs: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], Optional[str]]:
        """
        Computes granular category and item affinities based on customer demographics,
        persona archetype, and trained ML behavioral patterns.

        Domain Rules (Demographics & Behavioral Affinities):
        - Young / Youth demographic (Age < 26) or Tech/Gamer persona:
          Strong affinity boost for Laptops, PS5 / Gaming Consoles, 5G Smartphones, Tablets, Audio/Speakers.
        - Women / Homemaker demographic (Gender == 'Female' or Homemaker/Home Living persona):
          Strong affinity boost for Home Appliances: Washing Machine, Refrigerator, Microwave/Oven, Water Purifier, Vacuum Cleaner, Air Conditioner, Fan.
        - Working Professional (25-40): Productivity Laptops, Smartphones, Tablets, Air Conditioners.
        - Gaming Enthusiast: PS5, 4K TVs, Speakers, Gaming Laptops.
        """
        if not customer_profile:
            return ({}, None)

        raw_age = customer_profile.get("Age") if customer_profile.get("Age") is not None else customer_profile.get("age", 30)
        try:
            age = float(raw_age)
        except Exception:
            age = 30.0

        gender = str(customer_profile.get("Gender") or customer_profile.get("gender") or "").strip().lower()
        persona = str(customer_profile.get("persona") or customer_profile.get("Persona") or "").strip().lower()
        interests = customer_profile.get("interests", [])
        if isinstance(interests, str):
            interests = [interests.lower()]
        else:
            interests = [str(i).lower() for i in interests]

        affinities: Dict[str, float] = {}
        for cat in self.catalog_df["Category"].dropna().unique():
            affinities[cat] = 0.50
        for item in self.catalog_df["Item"].dropna().unique():
            affinities[item] = 0.50

        # Incorporate ML model predictions if available
        if ml_probs:
            for k, prob in ml_probs.items():
                affinities[k] = min(0.95, max(0.20, affinities.get(k, 0.50) * 0.4 + prob * 0.6))

        is_gamer = ("gamer" in persona) or ("ps5" in persona) or any("gaming" in i for i in interests)
        is_young = (age <= 25) or ("student" in persona) or ("young" in persona) or is_gamer
        is_woman_homemaker = (gender == "female") or ("women" in persona) or ("homemaker" in persona) or ("home" in persona and "appliance" in persona)
        is_pro = ("pro" in persona) or ("work" in persona) or ("developer" in persona)

        persona_summary = None

        if is_gamer or (is_young and ("game" in persona or "ps5" in persona)):
            persona_summary = "Youth & Gaming Persona: High preference for PS5, Gaming Gear, Laptops & 4K Entertainment"
            boosts = {
                "PS5": 0.99, "Gaming Console": 0.99, "Laptop": 0.92,
                "Television": 0.88, "Speaker": 0.85, "Mobile Phone": 0.82
            }
            for k, v in boosts.items():
                affinities[k] = v
        elif is_young:
            persona_summary = "Young Student / Gen-Z Persona: High preference for Laptops, PS5 Gaming, Smartphones & Audio"
            boosts = {
                "Laptop": 0.97, "PS5": 0.95, "Gaming Console": 0.95,
                "Mobile Phone": 0.90, "Tablet": 0.86, "Speaker": 0.82, "Charger": 0.75
            }
            for k, v in boosts.items():
                affinities[k] = v
        elif is_woman_homemaker:
            persona_summary = "Women & Home Care Persona: High preference for Home Appliances (Washing Machine, Refrigerator, Kitchen & Cleaning)"
            boosts = {
                "Washing Machine": 0.98,
                "Refrigerator": 0.96,
                "Oven": 0.94,
                "Water Purifier": 0.92,
                "Vacuum Cleaner": 0.90,
                "Air Conditioner": 0.85,
                "AC": 0.85,
                "Fan": 0.80,
                "Television": 0.72
            }
            for k, v in boosts.items():
                affinities[k] = v
        elif is_pro:
            persona_summary = "Working Professional Persona: High preference for Productivity Laptops, 5G Flagships, Tablets & AC"
            boosts = {
                "Laptop": 0.95, "Mobile Phone": 0.90, "Tablet": 0.86,
                "Air Conditioner": 0.82, "AC": 0.82, "Television": 0.75
            }
            for k, v in boosts.items():
                affinities[k] = v

        return affinities, persona_summary

    def recommend(
        self,
        query: str,
        customer_profile: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        strict_category: bool = True,
        strict_budget: bool = False,
        budget_override: Optional[float] = None,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes end-to-end hybrid recommendation workflow:
        1. Natural language query parsing
        2. Demographic & persona affinity computation
        3. Candidate filtering
        4. Multi-signal scoring (Semantic, Budget, Requirement, Discount, ML/Persona)
        5. Hybrid ranking & explainability generation
        """
        parsed = parse_query(query)
        if budget_override is not None:
            parsed["budget"] = float(budget_override)
        detected_category = parsed.get("category")
        user_budget = parsed.get("budget")
        keywords = parsed.get("keywords", [])

        # Compute semantic similarities for entire catalog
        sim_scores = self.semantic_engine.compute_similarity(query)

        # Compute ML category affinity if customer profile provided
        ml_cat_probs = {}
        if self.ml_model is not None and customer_profile:
            try:
                ml_cat_probs = self.ml_model.predict_category_affinity(customer_profile)
            except Exception:
                ml_cat_probs = {}

        # Compute demographic & persona affinity
        persona_affinities, persona_summary = self.compute_persona_affinity(customer_profile, ml_cat_probs)

        # Dynamic weights: if query is broad and customer profile is supplied, elevate behavioral ML weight
        active_weights = self.weights.copy()
        if (not detected_category or not keywords) and customer_profile:
            active_weights["behavioral_ml"] = max(0.35, active_weights.get("behavioral_ml", 0.10) * 2.5)
            total_w = sum(active_weights.values())
            active_weights = {k: v / total_w for k, v in active_weights.items()}

        # Candidate selection & scoring
        candidates = []
        for idx, row in self.catalog_df.iterrows():
            prod_cat = str(row.get("Category", ""))
            prod_item = str(row.get("Item", ""))
            price = float(row.get("Selling Price (INR)", 0))
            discount = float(row.get("Discount Rate (%)", 0))
            doc = str(row.get("Product Document", ""))
            suitable = str(row.get("Suitable For", ""))

            # Category filtering
            is_cat_match = True
            if detected_category:
                is_cat_match = (
                    detected_category.lower() in prod_cat.lower() or
                    detected_category.lower() in prod_item.lower() or
                    prod_cat.lower() in detected_category.lower() or
                    prod_item.lower() in detected_category.lower()
                )

            # Strict candidate filter check
            if strict_category and detected_category and not is_cat_match:
                continue

            if strict_budget and user_budget and (price > user_budget):
                continue

            # Individual Signal Calculations
            s_sem = float(sim_scores[idx])
            s_bud = self.semantic_engine.calculate_budget_fit(price, user_budget)
            s_req = self.semantic_engine.calculate_requirement_fit(doc, suitable, keywords)
            s_disc = min(1.0, discount / 35.0)  # normalized discount signal (0 to ~35%)
            
            # ML & Persona behavioral score
            s_ml = persona_affinities.get(
                prod_item,
                persona_affinities.get(
                    prod_cat,
                    ml_cat_probs.get(prod_item, ml_cat_probs.get(prod_cat, 0.50))
                )
            )

            # Hybrid Score formula
            hybrid_score = (
                active_weights["semantic"] * s_sem +
                active_weights["budget_fit"] * s_bud +
                active_weights["requirement_fit"] * s_req +
                active_weights["discount"] * s_disc +
                active_weights["behavioral_ml"] * s_ml
            )

            # Persona explanation note
            persona_note = None
            if persona_summary:
                if prod_item in ["Laptop", "PS5"] and ("Youth" in persona_summary or "Gamer" in persona_summary):
                    persona_note = "🎯 Top Pick for You: Aligns with young tech & gaming preferences (Laptops & PS5)"
                elif prod_item in ["Washing Machine", "Refrigerator", "Oven", "Vacuum Cleaner", "Water Purifier"] and "Women" in persona_summary:
                    persona_note = "🎯 Top Pick for You: Aligns with home appliances & household convenience"
                elif s_ml >= 0.85:
                    persona_note = f"🎯 Strongly matches your {persona_summary.split(':')[0]} preference"

            # Generate natural language explanation
            explanation = self.semantic_engine.generate_explanation(
                product=row,
                parsed_query=parsed,
                semantic_score=s_sem,
                budget_fit=s_bud,
                req_fit=s_req,
                persona_note=persona_note
            )

            candidates.append({
                "product_id": row.get("Product ID", f"PROD_{idx+1}"),
                "item": prod_item,
                "category": prod_cat,
                "specs": row.get("Product Specs", ""),
                "suitable_for": suitable,
                "mrp": float(row.get("MRP (INR)", price)),
                "selling_price": price,
                "discount_rate": discount,
                "semantic_score": s_sem,
                "budget_fit": s_bud,
                "requirement_fit": s_req,
                "discount_signal": s_disc,
                "ml_score": s_ml,
                "hybrid_score": hybrid_score,
                "explanation": explanation,
                "persona_note": persona_note,
                "is_category_match": is_cat_match,
                "is_within_budget": (user_budget is None) or (price <= user_budget)
            })

        # Fallback if strict filtering removed all candidates
        if not candidates:
            for idx, row in self.catalog_df.iterrows():
                price = float(row.get("Selling Price (INR)", 0))
                discount = float(row.get("Discount Rate (%)", 0))
                s_sem = float(sim_scores[idx])
                s_bud = self.semantic_engine.calculate_budget_fit(price, user_budget)
                s_req = self.semantic_engine.calculate_requirement_fit(str(row.get("Product Document", "")), str(row.get("Suitable For", "")), keywords)
                s_disc = min(1.0, discount / 35.0)
                s_ml = persona_affinities.get(str(row.get("Item", "")), 0.50)
                hybrid_score = (
                    active_weights["semantic"] * s_sem +
                    active_weights["budget_fit"] * s_bud +
                    active_weights["requirement_fit"] * s_req +
                    active_weights["discount"] * s_disc +
                    active_weights["behavioral_ml"] * s_ml
                )
                candidates.append({
                    "product_id": row.get("Product ID", f"PROD_{idx+1}"),
                    "item": row.get("Item", ""),
                    "category": row.get("Category", ""),
                    "specs": row.get("Product Specs", ""),
                    "suitable_for": row.get("Suitable For", ""),
                    "mrp": float(row.get("MRP (INR)", price)),
                    "selling_price": price,
                    "discount_rate": discount,
                    "semantic_score": s_sem,
                    "budget_fit": s_bud,
                    "requirement_fit": s_req,
                    "discount_signal": s_disc,
                    "ml_score": s_ml,
                    "hybrid_score": hybrid_score,
                    "explanation": self.semantic_engine.generate_explanation(row, parsed, s_sem, s_bud, s_req),
                    "persona_note": None,
                    "is_category_match": False,
                    "is_within_budget": (user_budget is None) or (price <= user_budget)
                })

        # Rank candidates descending by hybrid_score
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_recommendations = candidates[:top_k]

        return {
            "query": query,
            "parsed_query": parsed,
            "total_candidates": len(candidates),
            "recommendations": top_recommendations,
            "weights_used": active_weights,
            "persona_summary": persona_summary,
            "persona_affinities": persona_affinities
        }
