#!/usr/bin/env python3
"""Generate a synthetic dataset for mobile phone quarterly sales."""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    brands = ["Apple", "Samsung", "Xiaomi", "OnePlus"]
    os_choices = ["iOS", "Android"]

    rows = []
    for i in range(n):
        brand = rng.choice(brands)
        # iOS only for Apple, Android for others
        if brand == "Apple":
            os = "iOS"
        else:
            os = rng.choice(["Android"], p=[1.0])
        
        # Brand-specific pricing
        price_ranges = {
            "Apple": (900, 1400),      # Expensive
            "Samsung": (300, 900),     # Mid to Premium
            "Xiaomi": (150, 600),      # Budget to Mid
            "OnePlus": (250, 800),     # Mid to Premium
        }
        base_price, variance = price_ranges[brand]
        price = rng.normal(base_price, variance / 3)
        price = max(100, min(price, 1500))  # Clip to realistic range
        
        ram = rng.choice([2, 3, 4, 6, 8, 12], p=[0.05, 0.1, 0.2, 0.3, 0.25, 0.1])
        storage = rng.choice([32, 64, 128, 256], p=[0.1, 0.4, 0.35, 0.15])
        battery = rng.integers(3000, 5500)
        camera_mp = rng.choice([8, 12, 16, 48, 64, 108], p=[0.1, 0.15, 0.2, 0.25, 0.2, 0.1])
        promo = rng.choice([0, 1], p=[0.7, 0.3])
        sentiment = rng.normal(0, 1)
        quarter = rng.integers(1, 5)

        # base demand by brand (market share reflects real trends)
        base = {
            "Apple": 2500,      # Premium, high demand
            "Samsung": 2000,    # Large market share
            "Xiaomi": 1500,     # Growing market
            "OnePlus": 800,     # Niche/premium segment
        }[brand]

        demand = (
            base
            + (10 - price / 200) * 30   # Weaker price elasticity (was quadratic-dominant)
            + ram * 100                  # RAM matters significantly (8GB → +800)
            + storage * 3               # Storage matters (256GB → +768)
            + (battery - 3000) / 4      # Battery matters (5500mAh → +625)
            + camera_mp * 20            # Camera matters (108MP → +2160)
            + promo * 500               # Promotions have strong effect
            + sentiment * 150           # Sentiment has meaningful impact
            + rng.normal(0, 300)        # Realistic noise
        )

        revenue = max(demand, 0) * max(price, 50)

        rows.append(
            {
                "brand": brand,
                "os": os,
                "price": round(float(price), 2),
                "ram": int(ram),
                "storage": int(storage),
                "battery": int(battery),
                "camera_mp": int(camera_mp),
                "promo": int(promo),
                "sentiment": float(sentiment),
                "quarter": int(quarter),
                "sales_volume": max(int(demand), 0),
                "revenue": float(revenue),
            }
        )

    df = pd.DataFrame(rows)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(n=args.n, seed=args.seed)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
