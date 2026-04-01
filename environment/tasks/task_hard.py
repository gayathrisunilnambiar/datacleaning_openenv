"""
Hard Task — Hospital Admissions (200 rows × 10 columns).

Injected issues (all reproducible via seed 42):
  1. ~15 % of rows duplicated  (from easy)
  2. ~20 % of 'age' and 'weight_kg' set to NaN  (from easy)
  3. 'admission_date' has 3 mixed date formats  (from medium)
  4. 'gender' has inconsistent representations: M/F/Male/Female/male/female/0/1
  5. ~15 % of 'weight_kg' values converted from kg to lbs (mixed units)
  6. 'blood_type' has typos: A+, A_pos, Apos, a+, etc.
  7. 'readmitted' column stored as mixed yes/no/True/False/1/0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from environment.tasks.base_task import BaseTask


class HardTask(BaseTask):
    """Hospital-admissions cleaning task (difficulty: hard).

    Ground truth has 200 rows with columns:
    patient_id, admission_date, discharge_date, diagnosis, ward,
    age, gender, weight_kg, blood_type, readmitted.
    """

    task_id: str = "hard"
    difficulty: str = "hard"
    description: str = (
        "Hospital admissions with duplicates, missing values, mixed date "
        "formats, inconsistent gender labels, mixed weight units, "
        "blood-type typos, and heterogeneous boolean encoding."
    )

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    _DIAGNOSES: list[str] = [
        "Pneumonia", "Fracture", "Appendicitis", "Heart Failure",
        "Asthma", "Diabetes", "Stroke", "Migraine", "Infection",
        "Anemia",
    ]
    _WARDS: list[str] = ["ICU", "General", "Pediatrics", "Oncology", "Cardiology"]
    _BLOOD_TYPES_CLEAN: list[str] = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

    # Mapping: clean blood type → list of plausible dirty variants
    _BLOOD_TYPE_VARIANTS: dict[str, list[str]] = {
        "A+": ["A+", "A_pos", "Apos", "a+"],
        "A-": ["A-", "A_neg", "Aneg", "a-"],
        "B+": ["B+", "B_pos", "Bpos", "b+"],
        "B-": ["B-", "B_neg", "Bneg", "b-"],
        "AB+": ["AB+", "AB_pos", "ABpos", "ab+"],
        "AB-": ["AB-", "AB_neg", "ABneg", "ab-"],
        "O+": ["O+", "O_pos", "Opos", "o+"],
        "O-": ["O-", "O_neg", "Oneg", "o-"],
    }

    _GENDER_DIRTY_VARIANTS: list[str] = [
        "M", "F", "Male", "Female", "male", "female", "0", "1",
    ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_clean_df() -> pd.DataFrame:
        """Build the canonical 200-row hospital DataFrame (seed 42)."""
        rng = np.random.RandomState(42)
        n = 200

        admission_dates = pd.date_range("2023-01-01", periods=n, freq="2D")
        lengths_of_stay = rng.randint(1, 14, size=n)
        discharge_dates = admission_dates + pd.to_timedelta(lengths_of_stay, unit="D")

        genders = rng.choice(["Male", "Female"], size=n).tolist()

        df = pd.DataFrame(
            {
                "patient_id": [f"P-{i:04d}" for i in range(1, n + 1)],
                "admission_date": admission_dates.strftime("%Y-%m-%d").tolist(),
                "discharge_date": discharge_dates.strftime("%Y-%m-%d").tolist(),
                "diagnosis": rng.choice(HardTask._DIAGNOSES, size=n).tolist(),
                "ward": rng.choice(HardTask._WARDS, size=n).tolist(),
                "age": rng.randint(18, 90, size=n).astype(float).tolist(),
                "gender": genders,
                "weight_kg": np.round(rng.uniform(45.0, 120.0, size=n), 1).tolist(),
                "blood_type": rng.choice(HardTask._BLOOD_TYPES_CLEAN, size=n).tolist(),
                "readmitted": rng.choice([True, False], size=n).tolist(),
            }
        )
        df["age"] = df["age"].astype(float)
        df["weight_kg"] = df["weight_kg"].astype(float)
        return df

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_ground_truth_df(self) -> pd.DataFrame:
        """Return the clean hospital admissions DataFrame."""
        return self._build_clean_df()

    def get_dirty_df(self) -> pd.DataFrame:
        """Return the dirty version with all hard-level issues.

        Modifications applied (in order):
        1.  Duplicate ~15 % of rows.
        2.  Set ~20 % of 'age' to NaN.
        3.  Set ~20 % of 'weight_kg' to NaN.
        4.  Mix 3 date formats in 'admission_date'.
        5.  Replace 'gender' with inconsistent variants.
        6.  Convert ~15 % of non-NaN 'weight_kg' from kg → lbs.
        7.  Replace 'blood_type' with random typo variants.
        8.  Encode 'readmitted' as mixed yes/no/True/False/1/0.
        """
        rng = np.random.RandomState(42)
        df = self._build_clean_df()

        # --- 1. Duplicate rows (~15 %) ----------------------------------
        n_dup = int(len(df) * 0.15)
        dup_indices = rng.choice(df.index, size=n_dup, replace=False)
        df = pd.concat([df, df.loc[dup_indices]], ignore_index=True)

        # --- 2. NaN in 'age' (~20 %) ------------------------------------
        n_age_nan = int(len(df) * 0.20)
        age_nan_idx = rng.choice(df.index, size=n_age_nan, replace=False)
        df.loc[age_nan_idx, "age"] = np.nan

        # --- 3. NaN in 'weight_kg' (~20 %) ------------------------------
        n_wt_nan = int(len(df) * 0.20)
        wt_nan_idx = rng.choice(df.index, size=n_wt_nan, replace=False)
        df.loc[wt_nan_idx, "weight_kg"] = np.nan

        # --- 4. Mixed date formats in 'admission_date' -----------------
        original_dates = pd.to_datetime(df["admission_date"])
        new_dates: list[str] = []
        for i, dt in enumerate(original_dates):
            bucket = i % 3
            if bucket == 0:
                new_dates.append(dt.strftime("%Y-%m-%d"))        # 2023-01-15
            elif bucket == 1:
                new_dates.append(dt.strftime("%d/%m/%Y"))        # 15/01/2023
            else:
                new_dates.append(dt.strftime("%B %d %Y"))        # January 15 2023
        df["admission_date"] = new_dates

        # --- 5. Inconsistent gender labels ------------------------------
        gender_map_male = ["M", "Male", "male", "1"]
        gender_map_female = ["F", "Female", "female", "0"]

        new_gender: list[str] = []
        for i, g in enumerate(df["gender"]):
            if g == "Male":
                new_gender.append(gender_map_male[rng.randint(0, len(gender_map_male))])
            else:  # Female
                new_gender.append(gender_map_female[rng.randint(0, len(gender_map_female))])
        df["gender"] = new_gender

        # --- 6. Mixed weight units (~15 % in lbs) ----------------------
        non_nan_wt = df["weight_kg"].dropna().index.tolist()
        n_lbs = int(len(non_nan_wt) * 0.15)
        lbs_indices = rng.choice(non_nan_wt, size=n_lbs, replace=False)
        for idx in lbs_indices:
            kg_val = df.at[idx, "weight_kg"]
            df.at[idx, "weight_kg"] = round(kg_val * 2.20462, 1)  # kg → lbs

        # --- 7. Blood-type typos ----------------------------------------
        new_bt: list[str] = []
        for bt in df["blood_type"]:
            variants = self._BLOOD_TYPE_VARIANTS.get(bt, [bt])
            new_bt.append(variants[rng.randint(0, len(variants))])
        df["blood_type"] = new_bt

        # --- 8. Mixed readmitted encoding -------------------------------
        readmit_true_variants = ["yes", "True", "1", "Yes"]
        readmit_false_variants = ["no", "False", "0", "No"]
        new_readmit: list[str] = []
        for val in df["readmitted"]:
            if val is True or val == "True":
                new_readmit.append(
                    readmit_true_variants[rng.randint(0, len(readmit_true_variants))]
                )
            else:
                new_readmit.append(
                    readmit_false_variants[rng.randint(0, len(readmit_false_variants))]
                )
        df["readmitted"] = new_readmit

        return df

    def get_metadata(self) -> dict:
        """Return task metadata including column types and issue list."""
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "description": self.description,
            "column_types": {
                "patient_id": "object",
                "admission_date": "object",
                "discharge_date": "object",
                "diagnosis": "object",
                "ward": "object",
                "age": "float64",
                "gender": "object",
                "weight_kg": "float64",
                "blood_type": "object",
                "readmitted": "bool",
            },
            "num_rows": 200,
            "num_cols": 10,
            "issues": [
                "duplicate_rows",
                "missing_age",
                "missing_weight_kg",
                "mixed_date_formats",
                "inconsistent_gender",
                "mixed_weight_units_kg_lbs",
                "blood_type_typos",
                "mixed_boolean_readmitted",
            ],
        }


# ------------------------------------------------------------------
# Registry entry
# ------------------------------------------------------------------
TASK_REGISTRY: dict[str, type] = {"hard": HardTask}


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    task = HardTask()
    dirty = task.get_dirty_df()
    clean = task.get_ground_truth_df()
    meta = task.get_metadata()

    print("=== HARD TASK — Hospital Admissions ===")
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
    print(f"NaN counts in dirty:\n{dirty.isna().sum()}")
    print(f"Duplicate rows in dirty: {dirty.duplicated().sum()}")
    print(f"\nUnique gender values : {sorted(dirty['gender'].unique())}")
    print(f"Unique blood_type    : {sorted(dirty['blood_type'].unique())}")
    print(f"Unique readmitted    : {sorted(dirty['readmitted'].unique())}")
    print(f"\nWeight range (may have lbs): "
          f"min={dirty['weight_kg'].min():.1f}, max={dirty['weight_kg'].max():.1f}")
