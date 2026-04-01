"""
Medium Task — Sales Transactions (120 rows × 8 columns).

Injected issues (all reproducible via seed 42):
  1. 'quantity' stored as string with mixed formats ("  12 ", "7.00", etc.)
  2. 'unit_price' stored as string with dollar signs ("$15.5", "7.00")
  3. 'date' column has 3 mixed formats: YYYY-MM-DD, DD/MM/YYYY, Month DD YYYY
  4. 5 rows where total != quantity × unit_price
  5. ~10 % outliers in unit_price (values ≈ 10× the mean)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from environment.tasks.base_task import BaseTask


class MediumTask(BaseTask):
    """Sales-transactions cleaning task (difficulty: medium).

    Ground truth has 120 rows with columns:
    txn_id, date, product, category, quantity, unit_price, total, region.
    """

    task_id: str = "medium"
    difficulty: str = "medium"
    description: str = (
        "Sales transactions with mixed string formats in numeric columns, "
        "inconsistent date formats, arithmetic errors in 'total', "
        "and outliers in 'unit_price'."
    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_clean_df() -> pd.DataFrame:
        """Build the canonical 120-row sales DataFrame (seed 42)."""
        rng = np.random.RandomState(42)
        n = 120

        products = [
            "Widget A", "Widget B", "Gadget X", "Gadget Y",
            "Gizmo S", "Gizmo M", "Doohickey", "Thingamajig",
        ]
        categories = ["Electronics", "Hardware", "Accessories", "Software"]
        regions = ["North", "South", "East", "West"]

        dates = pd.date_range("2024-01-01", periods=n, freq="D")

        quantity = rng.randint(1, 50, size=n).astype(float)
        unit_price = np.round(rng.uniform(5.0, 80.0, size=n), 2)
        total = np.round(quantity * unit_price, 2)

        df = pd.DataFrame(
            {
                "txn_id": [f"TXN-{i:04d}" for i in range(1, n + 1)],
                "date": dates.strftime("%Y-%m-%d").tolist(),
                "product": rng.choice(products, size=n).tolist(),
                "category": rng.choice(categories, size=n).tolist(),
                "quantity": quantity,
                "unit_price": unit_price,
                "total": total,
                "region": rng.choice(regions, size=n).tolist(),
            }
        )
        return df

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_ground_truth_df(self) -> pd.DataFrame:
        """Return the clean sales DataFrame."""
        return self._build_clean_df()

    def get_dirty_df(self) -> pd.DataFrame:
        """Return the dirty version with format and arithmetic issues.

        Modifications applied (in order):
        1. Convert 'quantity' to strings with mixed whitespace/format.
        2. Convert 'unit_price' to strings with $ signs and padding.
        3. Mix 3 date formats across the 'date' column.
        4. Corrupt 'total' in 5 specific rows.
        5. Inject ~10 % outliers in 'unit_price' (multiply by ~10).
        """
        rng = np.random.RandomState(42)
        df = self._build_clean_df()

        # --- 1. Quantity → messy strings --------------------------------
        qty_formats = [
            lambda v: f"  {int(v)} ",            # padded integer
            lambda v: f"{v:.2f}",                 # float string
            lambda v: str(int(v)),                # plain integer
            lambda v: f" {int(v)}",               # left-padded
        ]
        new_qty: list[str] = []
        for i, val in enumerate(df["quantity"]):
            fmt = qty_formats[rng.randint(0, len(qty_formats))]
            new_qty.append(fmt(val))
        df["quantity"] = new_qty

        # --- 2. Unit_price → messy strings with $ ----------------------
        price_formats = [
            lambda v: f"${v:.2f}",                # $15.50
            lambda v: f"{v:.2f}",                  # 15.50
            lambda v: f"${v}",                     # $15.5
            lambda v: f"  {v:.1f} ",               # padded
        ]
        new_price: list[str] = []
        for i, val in enumerate(df["unit_price"]):
            fmt = price_formats[rng.randint(0, len(price_formats))]
            new_price.append(fmt(val))
        df["unit_price"] = new_price

        # --- 3. Mixed date formats --------------------------------------
        original_dates = pd.to_datetime(df["date"])
        new_dates: list[str] = []
        for i, dt in enumerate(original_dates):
            bucket = i % 3
            if bucket == 0:
                new_dates.append(dt.strftime("%Y-%m-%d"))        # 2024-01-15
            elif bucket == 1:
                new_dates.append(dt.strftime("%d/%m/%Y"))        # 15/01/2024
            else:
                new_dates.append(dt.strftime("%B %d %Y"))        # January 15 2024
        df["date"] = new_dates

        # --- 4. Arithmetic errors in 'total' (5 rows) ------------------
        error_indices = rng.choice(df.index, size=5, replace=False)
        for idx in error_indices:
            # Corrupt by adding a random offset
            df.at[idx, "total"] = round(df.at[idx, "total"] + rng.uniform(10, 100), 2)

        # --- 5. Outliers in unit_price (~10 %) --------------------------
        n_outliers = int(len(df) * 0.10)
        outlier_indices = rng.choice(df.index, size=n_outliers, replace=False)
        for idx in outlier_indices:
            raw = df.at[idx, "unit_price"]
            # Parse numeric value from the possibly-formatted string
            numeric_val = float(str(raw).replace("$", "").strip())
            inflated = round(numeric_val * 10, 2)
            df.at[idx, "unit_price"] = f"${inflated}"

        return df

    def get_metadata(self) -> dict:
        """Return task metadata including column types and issue list."""
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "description": self.description,
            "column_types": {
                "txn_id": "object",
                "date": "object",
                "product": "object",
                "category": "object",
                "quantity": "float64",
                "unit_price": "float64",
                "total": "float64",
                "region": "object",
            },
            "num_rows": 120,
            "num_cols": 8,
            "issues": [
                "quantity_as_messy_string",
                "unit_price_as_messy_string",
                "mixed_date_formats",
                "arithmetic_errors_in_total",
                "outliers_in_unit_price",
            ],
        }


# ------------------------------------------------------------------
# Registry entry
# ------------------------------------------------------------------
TASK_REGISTRY: dict[str, type] = {"medium": MediumTask}


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    task = MediumTask()
    dirty = task.get_dirty_df()
    clean = task.get_ground_truth_df()
    meta = task.get_metadata()

    print("=== MEDIUM TASK — Sales Transactions ===")
    print(f"Clean shape : {clean.shape}")
    print(f"Dirty shape : {dirty.shape}")
    print(f"Metadata    : {meta}")
    print()
    print("--- Clean head ---")
    print(clean.head(8).to_string(index=False))
    print()
    print("--- Dirty head ---")
    print(dirty.head(8).to_string(index=False))
    print()
    print("--- Dirty dtypes ---")
    print(dirty.dtypes)
    print()
    print(f"Sample dirty quantity values : {dirty['quantity'].head(5).tolist()}")
    print(f"Sample dirty unit_price values: {dirty['unit_price'].head(5).tolist()}")
    print(f"Sample dirty date values      : {dirty['date'].head(6).tolist()}")
