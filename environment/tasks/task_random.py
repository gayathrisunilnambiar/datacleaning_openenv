"""
Random Task — Procedurally generated dirty DataFrame.

Generates a clean DataFrame from one of several domain templates,
then injects 2-4 data quality issues chosen randomly.  Each episode
is fully reproducible when given the same seed.
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from environment.tasks.base_task import BaseTask


class RandomTask(BaseTask):
    """
    Procedurally generates a dirty DataFrame each time, seeded by the
    caller so episodes are reproducible when needed but varied across runs.
    """

    DOMAINS = [
        {
            "name": "employee_records",
            "columns": {
                "id": "int", "name": "str", "age": "int",
                "department": "str", "salary": "float", "join_date": "date",
            },
            "ranges": {"age": (18, 65), "salary": (20000, 200000)},
            "categories": {"department": ["HR", "Engineering", "Sales", "Marketing", "Finance"]},
        },
        {
            "name": "product_inventory",
            "columns": {
                "sku": "str", "product_name": "str", "category": "str",
                "price": "float", "stock": "int", "last_updated": "date",
            },
            "ranges": {"price": (1.0, 9999.0), "stock": (0, 10000)},
            "categories": {"category": ["Electronics", "Clothing", "Food", "Books", "Tools"]},
        },
        {
            "name": "customer_orders",
            "columns": {
                "order_id": "int", "customer_name": "str", "product": "str",
                "quantity": "int", "unit_price": "float", "order_date": "date",
                "status": "str",
            },
            "ranges": {"quantity": (1, 100), "unit_price": (5.0, 500.0)},
            "categories": {"status": ["pending", "shipped", "delivered", "cancelled"]},
        },
    ]

    ISSUE_TYPES = [
        "duplicate_rows",
        "missing_numeric",
        "missing_categorical",
        "type_as_string",
        "mixed_dates",
        "outliers",
        "inconsistent_categorical",
    ]

    # ── Realistic name pools ──────────────────────────────────────────

    _FIRST_NAMES = [
        "Alice", "Bob", "Carlos", "Diana", "Eve", "Frank", "Grace", "Hector",
        "Irene", "Jack", "Karen", "Leo", "Mia", "Nate", "Olivia", "Paul",
        "Quinn", "Rita", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
        "Yara", "Zane", "Aiden", "Beth", "Caleb", "Donna", "Eli", "Faye",
        "Gus", "Holly", "Ivan", "Jill", "Kyle", "Luna", "Mike", "Nora",
        "Oscar", "Penny", "Reed", "Sara", "Troy", "Ursula", "Vince", "Willa",
    ]

    _LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
        "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
        "Wright", "Scott", "Torres", "Hill", "Green", "Adams", "Baker",
    ]

    _PRODUCT_NAMES: dict[str, list[str]] = {
        "Electronics": [
            "Wireless Headphones", "USB-C Charger", "Bluetooth Speaker",
            "HDMI Cable", "Portable SSD", "Webcam Pro", "LED Monitor",
            "Mechanical Keyboard", "Gaming Mouse", "Power Bank",
        ],
        "Clothing": [
            "Cotton T-Shirt", "Denim Jacket", "Slim Fit Jeans",
            "Wool Sweater", "Running Shoes", "Baseball Cap",
            "Linen Shirt", "Cargo Pants", "Winter Coat", "Silk Scarf",
        ],
        "Food": [
            "Organic Honey", "Dark Chocolate Bar", "Olive Oil Extra Virgin",
            "Protein Powder", "Green Tea Pack", "Almond Butter",
            "Granola Mix", "Coconut Water", "Trail Mix", "Energy Bars",
        ],
        "Books": [
            "Python Cookbook", "Data Science Handbook", "ML Guide",
            "Clean Code", "Design Patterns", "Algorithm Primer",
            "Web Dev Bootcamp", "Database Internals", "Cloud Architecture",
            "DevOps Handbook",
        ],
        "Tools": [
            "Cordless Drill", "Wrench Set", "Digital Multimeter",
            "Soldering Iron", "Tape Measure", "Utility Knife",
            "Pliers Set", "Screwdriver Kit", "Safety Goggles", "Work Gloves",
        ],
    }

    _GENERIC_PRODUCTS = [
        "Widget Alpha", "Widget Beta", "Gadget Pro", "Gadget Lite",
        "Module X", "Module Y", "Component A", "Accessory Pack",
    ]

    # ── Constructor ───────────────────────────────────────────────────

    def __init__(self, seed: int | None = None):
        self.seed = seed if seed is not None else random.randint(0, 999_999)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.RandomState(self.seed)

        # Randomly select domain and 2-4 issue types
        self.domain = self.rng.choice(self.DOMAINS)
        n_issues = self.rng.randint(2, 4)
        self.selected_issues = self.rng.sample(self.ISSUE_TYPES, n_issues)

        # BaseTask attributes (instance-level for dynamic content)
        self.task_id: str = "random"
        self.difficulty: str = "variable"
        self.max_steps: int = 25
        self.description: str = (
            f"Procedural task: {self.domain['name']} with "
            f"{', '.join(self.selected_issues)}"
        )

        # Generate data
        self._issue_log: dict[str, dict] = {}
        self._clean_df = self._generate_clean_df(n_rows=self.rng.randint(60, 150))
        self._dirty_df = self._inject_issues(self._clean_df.copy())

    # ── Clean data generators (per domain) ────────────────────────────

    def _generate_clean_df(self, n_rows: int) -> pd.DataFrame:
        gen = {
            "employee_records": self._gen_employee_records,
            "product_inventory": self._gen_product_inventory,
            "customer_orders": self._gen_customer_orders,
        }
        return gen[self.domain["name"]](n_rows)

    def _random_names(self, n: int) -> list[str]:
        return [
            f"{self.rng.choice(self._FIRST_NAMES)} {self.rng.choice(self._LAST_NAMES)}"
            for _ in range(n)
        ]

    def _random_dates(
        self, n: int, start_year: int = 2020, end_year: int = 2025,
    ) -> list[str]:
        start = datetime(start_year, 1, 1)
        span = (datetime(end_year, 12, 31) - start).days
        return [
            (start + timedelta(days=self.rng.randint(0, span))).strftime("%Y-%m-%d")
            for _ in range(n)
        ]

    def _gen_employee_records(self, n: int) -> pd.DataFrame:
        r = self.domain["ranges"]
        cats = self.domain["categories"]
        return pd.DataFrame({
            "id": list(range(1, n + 1)),
            "name": self._random_names(n),
            "age": self.np_rng.randint(r["age"][0], r["age"][1] + 1, size=n).astype(float).tolist(),
            "department": [self.rng.choice(cats["department"]) for _ in range(n)],
            "salary": self.np_rng.uniform(r["salary"][0], r["salary"][1], size=n).round(2).tolist(),
            "join_date": self._random_dates(n, 2018, 2025),
        })

    def _gen_product_inventory(self, n: int) -> pd.DataFrame:
        r = self.domain["ranges"]
        cats = self.domain["categories"]
        skus = [f"SKU-{self.rng.randint(10000, 99999)}" for _ in range(n)]
        products = []
        for _ in range(n):
            cat = self.rng.choice(cats["category"])
            pool = self._PRODUCT_NAMES.get(cat, self._GENERIC_PRODUCTS)
            products.append(self.rng.choice(pool))
        return pd.DataFrame({
            "sku": skus,
            "product_name": products,
            "category": [self.rng.choice(cats["category"]) for _ in range(n)],
            "price": self.np_rng.uniform(r["price"][0], r["price"][1], size=n).round(2).tolist(),
            "stock": self.np_rng.randint(r["stock"][0], r["stock"][1] + 1, size=n).tolist(),
            "last_updated": self._random_dates(n, 2023, 2025),
        })

    def _gen_customer_orders(self, n: int) -> pd.DataFrame:
        r = self.domain["ranges"]
        cats = self.domain["categories"]
        products = []
        for _ in range(n):
            cat_key = self.rng.choice(list(self._PRODUCT_NAMES.keys()))
            products.append(self.rng.choice(self._PRODUCT_NAMES[cat_key]))
        return pd.DataFrame({
            "order_id": list(range(1001, 1001 + n)),
            "customer_name": self._random_names(n),
            "product": products,
            "quantity": self.np_rng.randint(r["quantity"][0], r["quantity"][1] + 1, size=n).tolist(),
            "unit_price": self.np_rng.uniform(r["unit_price"][0], r["unit_price"][1], size=n).round(2).tolist(),
            "order_date": self._random_dates(n, 2023, 2025),
            "status": [self.rng.choice(cats["status"]) for _ in range(n)],
        })

    # ── Issue injection ───────────────────────────────────────────────

    def _inject_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        self._issue_log = {}
        col_types = self.domain["columns"]

        # Identify the key column the grader uses for alignment — must not
        # be mutated by issue injection or scoring will break.
        key_col = None
        for c in col_types:
            if c == "id" or c.endswith("_id"):
                key_col = c
                break
        if key_col is None:
            key_col = next(iter(col_types))

        numeric_cols = [c for c, t in col_types.items() if t in ("int", "float") and c != key_col]
        str_cols = [c for c, t in col_types.items() if t == "str" and c != key_col]
        cat_cols = [
            c for c, t in col_types.items()
            if t == "str" and c in self.domain.get("categories", {}) and c != key_col
        ]
        date_cols = [c for c, t in col_types.items() if t == "date" and c != key_col]

        used: set[str] = set()

        _DISPATCH = {
            "duplicate_rows": (lambda: self._inject_duplicates(df), None),
            "missing_numeric": (lambda c: self._inject_missing_numeric(df, c), numeric_cols),
            "missing_categorical": (lambda c: self._inject_missing_categorical(df, c), str_cols),
            "type_as_string": (lambda c: self._inject_type_as_string(df, c), numeric_cols),
            "mixed_dates": (lambda c: self._inject_mixed_dates(df, c), date_cols),
            "outliers": (lambda c: self._inject_outliers(df, c), numeric_cols),
            "inconsistent_categorical": (lambda c: self._inject_inconsistent_cat(df, c), cat_cols),
        }

        for issue in self.selected_issues:
            fn, col_pool = _DISPATCH[issue]
            if col_pool is None:
                # row-level issue (duplicate_rows)
                df = fn()
            else:
                available = [c for c in col_pool if c not in used]
                if not available:
                    available = list(col_pool)  # allow reuse if exhausted
                if available:
                    col = self.rng.choice(available)
                    used.add(col)
                    df = fn(col)
        return df

    def _inject_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        frac = self.rng.uniform(0.10, 0.15)
        n_dup = max(1, int(len(df) * frac))
        indices = self.rng.sample(range(len(df)), min(n_dup, len(df)))
        df = pd.concat([df, df.iloc[indices]], ignore_index=True)
        self._issue_log["duplicate_rows"] = {"column": "__all__", "affected_rows": n_dup}
        return df

    def _inject_missing_numeric(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        n = max(1, int(len(df) * 0.20))
        idx = self.rng.sample(range(len(df)), min(n, len(df)))
        df[col] = df[col].astype(float)
        df.loc[idx, col] = np.nan
        self._issue_log["missing_numeric"] = {"column": col, "affected_rows": len(idx)}
        return df

    def _inject_missing_categorical(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        n = max(1, int(len(df) * 0.15))
        idx = self.rng.sample(range(len(df)), min(n, len(df)))
        df.loc[idx, col] = np.nan
        self._issue_log["missing_categorical"] = {"column": col, "affected_rows": len(idx)}
        return df

    def _inject_type_as_string(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        templates = [
            lambda v: f"${v:,.2f}",
            lambda v: f"{v:,}",
            lambda v: f" {v} ",
            lambda v: f"${v}",
            lambda v: f"{v:.1f}",
        ]
        new_vals, affected = [], 0
        for val in df[col]:
            if pd.isna(val):
                new_vals.append(val)
            else:
                new_vals.append(self.rng.choice(templates)(float(val)))
                affected += 1
        df[col] = new_vals
        self._issue_log["type_as_string"] = {"column": col, "affected_rows": affected}
        return df

    def _inject_mixed_dates(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        fmts = [
            lambda d: d,  # YYYY-MM-DD (keep original)
            lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m/%Y"),
            lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%B %d %Y"),
        ]
        new_vals, affected = [], 0
        for val in df[col]:
            if pd.isna(val):
                new_vals.append(val)
            else:
                fn = self.rng.choice(fmts)
                try:
                    new_vals.append(fn(str(val)))
                    if fn is not fmts[0]:
                        affected += 1
                except Exception:
                    new_vals.append(val)
        df[col] = new_vals
        self._issue_log["mixed_dates"] = {"column": col, "affected_rows": affected}
        return df

    def _inject_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        numeric = pd.to_numeric(df[col], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            return df
        col_mean = abs(float(valid.mean())) or 1.0
        n_out = max(1, int(len(df) * 0.05))
        idx = self.rng.sample(range(len(df)), min(n_out, len(df)))
        for i in idx:
            df.at[i, col] = round(col_mean * self.rng.uniform(4.0, 6.0), 2)
        self._issue_log["outliers"] = {"column": col, "affected_rows": len(idx)}
        return df

    def _inject_inconsistent_cat(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        n = max(1, int(len(df) * 0.20))
        idx = self.rng.sample(range(len(df)), min(n, len(df)))
        for i in idx:
            val = df.at[i, col]
            if pd.isna(val):
                continue
            s = str(val)
            transforms = [s.upper(), s.lower(), s.title() + " ", " " + s]
            df.at[i, col] = self.rng.choice(transforms)
        self._issue_log["inconsistent_categorical"] = {"column": col, "affected_rows": len(idx)}
        return df

    # ── Public interface ──────────────────────────────────────────────

    def get_dirty_df(self) -> pd.DataFrame:
        return self._dirty_df.copy()

    def get_ground_truth_df(self) -> pd.DataFrame:
        return self._clean_df.copy()

    def get_metadata(self) -> dict:
        clean = self._clean_df
        return {
            "task_id": "random",
            "seed": self.seed,
            "difficulty": "variable",
            "domain": self.domain["name"],
            "selected_issues": self.selected_issues,
            "description": self.description,
            "max_steps": self.max_steps,
            "issue_log": self._issue_log,
            "column_types": {col: str(dtype) for col, dtype in clean.dtypes.items()},
            "num_rows": len(clean),
            "num_cols": len(clean.columns),
            "issues": self.selected_issues,
        }


# ── Registry entry ────────────────────────────────────────────────
TASK_REGISTRY: dict[str, type] = {"random": RandomTask}


# ── Smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RANDOM TASK — Smoke Test (3 seeds)")
    print("=" * 60)

    for seed in [42, 123, 999]:
        task = RandomTask(seed=seed)
        dirty = task.get_dirty_df()
        clean = task.get_ground_truth_df()
        meta = task.get_metadata()

        print(f"\n--- Seed {seed} ---")
        print(f"Domain     : {meta['domain']}")
        print(f"Issues     : {meta['selected_issues']}")
        print(f"Clean shape: {clean.shape}")
        print(f"Dirty shape: {dirty.shape}")
        print(f"Issue log  : {meta['issue_log']}")
        print(f"\nDirty head:\n{dirty.head(5).to_string(index=False)}")
        print()
