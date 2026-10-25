"""
Query Parser Module
Extracts product category, numerical budget constraints, and specification keywords
from free-form natural language customer queries.
"""

import re
from typing import Dict, Any, Optional, List

# Category keyword and synonym mapping
CATEGORY_MAP = {
    "Laptop": [
        "laptop", "notebook", "laptops", "pc", "macbook", "ultrabook", "computer"
    ],
    "Washing Machine": [
        "washing machine", "washer", "dryer", "laundry", "front load", "top load", "washing"
    ],
    "Refrigerator": [
        "refrigerator", "fridge", "freezer", "double door", "single door", "frost free", "refrigerators"
    ],
    "Mobile Phone": [
        "mobile phone", "mobile", "phone", "smartphone", "cellphone", "android", "iphone", "handset", "5g phone"
    ],
    "Tablet": [
        "tablet", "tab", "ipad", "pad", "tablets"
    ],
    "Television": [
        "television", "tv", "smart tv", "led tv", "oled", "qled", "4k tv", "tvs", "screen"
    ],
    "Air Conditioner": [
        "air conditioner", "ac", "split ac", "window ac", "cooling", "inverter ac"
    ],
    "Fan": [
        "fan", "ceiling fan", "bldc fan", "table fan", "fans"
    ],
    "Gaming Console": [
        "ps5", "playstation", "gaming console", "console", "playstation 5", "xbox"
    ],
    "Speaker": [
        "speaker", "speakers", "bluetooth speaker", "party speaker", "soundbar", "audio", "sound"
    ],
    "Oven": [
        "oven", "microwave", "microwave oven", "convection oven", "solo microwave", "otg", "baking oven"
    ],
    "Vacuum Cleaner": [
        "vacuum cleaner", "vacuum", "vac", "dust cleaner", "cleaner", "cordless vacuum", "stick vacuum"
    ],
    "Water Purifier": [
        "water purifier", "purifier", "ro", "uv", "water filter", "ro purifier", "drinking water"
    ],
    "Charger": [
        "charger", "adapter", "fast charger", "gan charger", "usb-c charger", "power adapter", "charging"
    ]
}


def extract_budget(query: str) -> Optional[float]:
    """
    Extracts numerical budget from queries like:
    - 'under 60000', 'below ₹50,000', 'budget 40k', 'within 1.5 lakh', 'under 75k', '50k'
    """
    q = query.lower().replace(",", "")

    # Pattern for Lakh (e.g. 1.5 lakh, 1 lakh, 2.5 lakhs)
    lakh_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs)', q)
    if lakh_match:
        try:
            return float(lakh_match.group(1)) * 100000
        except ValueError:
            pass

    # Pattern for 'k' / 'thousand' (e.g. 60k, 50 k, 45 thousand)
    k_match = re.search(r'(?:under|below|budget|within|around|less than|upto|up to|max)?\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:k|thousand)', q)
    if k_match:
        try:
            return float(k_match.group(1)) * 1000
        except ValueError:
            pass

    # Explicit numbers with budget indicator words (e.g. under 60000, below 50000, budget 40000)
    budget_patterns = [
        r'(?:under|below|budget|within|around|less than|upto|up to|max|approx|price)\s*(?:₹|rs\.?|inr)?\s*(\d{3,7})',
        r'(?:₹|rs\.?|inr)\s*(\d{3,7})',
        r'(\d{4,7})\s*(?:budget|inr|rs|rupees)'
    ]

    for pat in budget_patterns:
        match = re.search(pat, q)
        if match:
            try:
                val = float(match.group(1))
                # Validate reasonable electronics price range (₹500 to ₹5,00,000)
                if 500 <= val <= 500000:
                    return val
            except ValueError:
                pass

    return None


def detect_category(query: str) -> Optional[str]:
    """
    Detects the product category from the customer query based on keyword mappings.
    Returns standardized category name or None if general/unmatched.
    """
    q_lower = query.lower()

    # Match longest phrases first to avoid false positives (e.g. "washing machine" before "machine")
    matches = []
    for standard_cat, synonyms in CATEGORY_MAP.items():
        for syn in synonyms:
            # Word boundary regex search
            pattern = r'\b' + re.escape(syn) + r'\b'
            if re.search(pattern, q_lower):
                matches.append((len(syn), standard_cat))

    if matches:
        # Pick the most specific (longest keyword) match
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[0][1]

    return None


def extract_keywords(query: str) -> List[str]:
    """Extracts key requirement features and technical terms."""
    q_lower = query.lower()
    technical_keywords = [
        "gaming", "coding", "python", "sql", "data analysis", "programming", "office", "student", "college",
        "16gb", "8gb", "32gb", "4gb", "12gb", "512gb", "1tb", "256gb", "ssd", "ram", "i5", "i7", "i3", "ryzen",
        "5g", "camera", "50mp", "108mp", "battery", "fast charging", "oled", "4k", "hdr", "dolby", "smart tv",
        "inverter", "5 star", "3 star", "1.5 ton", "1 ton", "2 ton", "frost free", "double door", "single door",
        "front load", "top load", "steam wash", "bldc", "remote", "hepa", "cordless", "pet", "pets",
        "ro", "uv", "mineralizer", "convection", "grill", "baking", "portable", "bass", "party", "gan", "fast charger"
    ]
    
    extracted = []
    for kw in technical_keywords:
        if kw in q_lower:
            extracted.append(kw)
    return extracted


def parse_query(query: str) -> Dict[str, Any]:
    """
    Complete natural language query parsing pipeline.
    Returns dictionary with raw query, detected category, budget, and extracted specs.
    """
    category = detect_category(query)
    budget = extract_budget(query)
    keywords = extract_keywords(query)

    return {
        "raw_query": query,
        "category": category,
        "budget": budget,
        "keywords": keywords,
        "has_category": category is not None,
        "has_budget": budget is not None
    }
