"""Final grader for the hard hospital-admissions task."""

from __future__ import annotations

import pandas as pd

from environment.graders.base_grader import BaseGrader
from environment.tasks.task_hard import HardTask


class HardGrader(BaseGrader):
    """Grader for multi-issue healthcare cleaning with mixed units and labels."""

    task_id = "hard"

    def __init__(self) -> None:
        super().__init__(HardTask())

    @staticmethod
    def _age_dtype_ready(series: pd.Series) -> float:
        if pd.api.types.is_integer_dtype(series):
            return 1.0
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all() and ((numeric % 1) == 0).all():
            return 1.0
        return 0.0

    def score(self, df: pd.DataFrame) -> float:
        scores = self.column_scores(df)

        row_count_score = 1.0 if len(df) == len(self.truth_df) else 0.0

        age_raw = self._series_from_column(df, "age")
        age_alignment_score = scores.get("age", 0.0)
        age_dtype_score = self._age_dtype_ready(age_raw)
        age_score = (0.5 * age_alignment_score) + (0.5 * age_dtype_score)

        admission_series = self._series_from_column(df, "admission_date")
        admission_score = (
            1.0 if pd.api.types.is_datetime64_any_dtype(admission_series) else self._iso_date_fraction(admission_series)
        )

        gender_raw = self._series_from_column(df, "gender").astype(str).str.strip()
        gender_score = self._fraction_valid(gender_raw.isin(["Male", "Female"]))

        blood_type_raw = self._series_from_column(df, "blood_type").astype(str).str.strip()
        blood_type_score = self._fraction_valid(
            blood_type_raw.isin(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
        )

        readmitted_raw = self._series_from_column(df, "readmitted")
        readmitted_score = 1.0 if pd.api.types.is_bool_dtype(readmitted_raw) else 0.0

        total_score = (
            (0.15 * row_count_score)
            + (0.10 * age_score)
            + (0.20 * admission_score)
            + (0.15 * gender_score)
            + (0.10 * scores.get("weight_kg", 0.0))
            + (0.15 * blood_type_score)
            + (0.15 * readmitted_score)
        )
        return self._clamp(total_score)


# ---------------------------------------------------------------------------
# Standalone diagnostic helper — does NOT modify any grading logic above
# ---------------------------------------------------------------------------

def diagnose(cleaned_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> None:
    """Print a detailed per-check breakdown showing exact score and failure reason.

    Covers every sub-check that HardGrader.score() uses:
      row_count, age (alignment + dtype), admission_date, gender,
      weight_kg (column_scores), blood_type, readmitted.
    """
    grader = HardGrader()
    col_scores = grader.column_scores(cleaned_df)
    truth = ground_truth_df

    SEP = "─" * 72

    def _header(name: str, weight: float, actual: float) -> None:
        pct = actual * weight
        print(f"\n{SEP}")
        print(f"CHECK : {name}")
        print(f"WEIGHT: {weight:.2f}  |  RAW SCORE: {actual:.4f}  |  CONTRIBUTION: {pct:.4f}")

    def _show_failures(mask: pd.Series, cleaned_sub: pd.DataFrame,
                       truth_sub: pd.DataFrame, cols: list[str], label: str = "Failed rows") -> None:
        failed_idx = mask[mask].index[:5]
        if len(failed_idx) == 0:
            print(f"  ✓ No failures found for {label}")
            return
        print(f"  ✗ {label} — showing up to 5 sample failures:")
        for idx in failed_idx:
            found_vals = {c: cleaned_sub.loc[idx, c] if c in cleaned_sub.columns else "<missing>"
                         for c in cols}
            truth_vals = {c: truth_sub.loc[idx, c] if c in truth_sub.columns else "<missing>"
                         for c in cols}
            print(f"    idx={idx}  FOUND={found_vals}  EXPECTED={truth_vals}")

    # Align frames the same way the grader does
    current_aligned, truth_aligned = grader._aligned_frames(cleaned_df)

    # ── 1. Row count ────────────────────────────────────────────────────────
    row_count_score = 1.0 if len(cleaned_df) == len(truth) else 0.0
    _header("row_count", 0.15, row_count_score)
    print(f"  Cleaned rows: {len(cleaned_df)}  |  Expected rows: {len(truth)}")
    if row_count_score < 1.0:
        diff = len(cleaned_df) - len(truth)
        print(f"  ✗ Row count mismatch: {'+' if diff > 0 else ''}{diff} rows vs ground truth")
    else:
        print("  ✓ Row count matches ground truth")

    # ── 2. Age — alignment + dtype ──────────────────────────────────────────
    age_alignment = col_scores.get("age", 0.0)
    age_series = grader._series_from_column(cleaned_df, "age")
    age_dtype_score = HardGrader._age_dtype_ready(age_series)
    age_score = 0.5 * age_alignment + 0.5 * age_dtype_score
    _header("age (alignment×0.5 + dtype×0.5)", 0.10, age_score)
    print(f"  Sub-scores: alignment={age_alignment:.4f}, dtype_ready={age_dtype_score:.4f}")
    if age_dtype_score < 1.0:
        print(f"  ✗ age dtype is '{age_series.dtype}' — expected integer dtype")
    if age_alignment < 1.0 and "age" in current_aligned.columns and "age" in truth_aligned.columns:
        cur_age = pd.to_numeric(current_aligned["age"], errors="coerce")
        tru_age = pd.to_numeric(truth_aligned["age"], errors="coerce")
        bad = (cur_age - tru_age).abs() > 0.5
        _show_failures(bad, current_aligned[["age"]], truth_aligned[["age"]], ["age"],
                       "age value mismatch")

    # ── 3. Admission date ───────────────────────────────────────────────────
    adm_series = grader._series_from_column(cleaned_df, "admission_date")
    if pd.api.types.is_datetime64_any_dtype(adm_series):
        admission_score = 1.0
    else:
        admission_score = grader._iso_date_fraction(adm_series)
    _header("admission_date", 0.20, admission_score)
    if pd.api.types.is_datetime64_any_dtype(adm_series):
        print("  ✓ Column is datetime64 dtype")
    else:
        print(f"  ✗ Column dtype is '{adm_series.dtype}' (not datetime64)")
        non_iso = adm_series.astype(str).str.strip()
        parsed = pd.to_datetime(non_iso, format="%Y-%m-%d", errors="coerce")
        bad_mask = parsed.isna() & adm_series.notna()
        bad_samples = adm_series[bad_mask].head(5)
        if not bad_samples.empty:
            print(f"  ✗ Non-ISO values (up to 5): {bad_samples.tolist()}")

    # ── 4. Gender ────────────────────────────────────────────────────────────
    gender_raw = grader._series_from_column(cleaned_df, "gender").astype(str).str.strip()
    valid_genders = {"Male", "Female"}
    gender_valid_mask = gender_raw.isin(valid_genders)
    gender_score = grader._fraction_valid(gender_valid_mask)
    _header("gender", 0.15, gender_score)
    if gender_score < 1.0:
        bad_gender = gender_raw[~gender_valid_mask]
        invalid_vals = bad_gender.value_counts().head(10)
        print(f"  ✗ {(~gender_valid_mask).sum()} rows with invalid gender values:")
        for val, cnt in invalid_vals.items():
            print(f"      '{val}': {cnt} occurrences")
        sample_idx = bad_gender.index[:5]
        for idx in sample_idx:
            print(f"    idx={idx}  FOUND='{gender_raw[idx]}'  EXPECTED='Male' or 'Female'")
    else:
        print("  ✓ All gender values are 'Male' or 'Female'")

    # ── 5. Weight_kg (column similarity score) ──────────────────────────────
    weight_score = col_scores.get("weight_kg", 0.0)
    _header("weight_kg", 0.10, weight_score)
    if weight_score < 1.0 and "weight_kg" in current_aligned.columns and "weight_kg" in truth_aligned.columns:
        cur_wt = pd.to_numeric(current_aligned["weight_kg"], errors="coerce")
        tru_wt = pd.to_numeric(truth_aligned["weight_kg"], errors="coerce")
        val_range = float(tru_wt.dropna().max() - tru_wt.dropna().min()) or 1.0
        abs_err = (cur_wt - tru_wt).abs()
        bad = abs_err > (0.05 * val_range)  # more than 5 % of range off
        print(f"  Approx rows diverging from truth (>5% range): {bad.sum()}")
        sample_bad = abs_err[bad].nlargest(5).index
        for idx in sample_bad:
            print(f"    idx={idx}  FOUND={cur_wt.get(idx, '?'):.1f}  EXPECTED={tru_wt.get(idx, '?'):.1f}"
                  f"  diff={abs_err.get(idx, 0):.1f}")
    else:
        print("  ✓ weight_kg is fully aligned with ground truth")

    # ── 6. Blood type ────────────────────────────────────────────────────────
    bt_raw = grader._series_from_column(cleaned_df, "blood_type").astype(str).str.strip()
    valid_bt = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}
    bt_valid_mask = bt_raw.isin(valid_bt)
    blood_type_score = grader._fraction_valid(bt_valid_mask)
    _header("blood_type", 0.15, blood_type_score)
    if blood_type_score < 1.0:
        bad_bt = bt_raw[~bt_valid_mask]
        invalid_bt_vals = bad_bt.value_counts().head(10)
        print(f"  ✗ {(~bt_valid_mask).sum()} rows with non-canonical blood_type:")
        for val, cnt in invalid_bt_vals.items():
            print(f"      '{val}': {cnt} occurrences")
    else:
        print("  ✓ All blood_type values are canonical (A+/A-/B+/B-/AB+/AB-/O+/O-)")

    # ── 7. Readmitted ────────────────────────────────────────────────────────
    readmitted_raw = grader._series_from_column(cleaned_df, "readmitted")
    readmitted_score = 1.0 if pd.api.types.is_bool_dtype(readmitted_raw) else 0.0
    _header("readmitted", 0.15, readmitted_score)
    if readmitted_score < 1.0:
        print(f"  ✗ readmitted dtype is '{readmitted_raw.dtype}' — expected bool dtype")
        unique_vals = readmitted_raw.unique()[:10]
        print(f"  Unique values found: {unique_vals.tolist()}")
    else:
        print("  ✓ readmitted is bool dtype")

    # ── Overall ──────────────────────────────────────────────────────────────
    total = (
        0.15 * row_count_score
        + 0.10 * age_score
        + 0.20 * admission_score
        + 0.15 * gender_score
        + 0.10 * weight_score
        + 0.15 * blood_type_score
        + 0.15 * readmitted_score
    )
    total = max(0.0, min(1.0, total))
    print(f"\n{SEP}")
    print(f"OVERALL SCORE : {total:.4f}")
    print(SEP)
    print("\nPer-check breakdown:")
    breakdown = {
        "row_count":       {"weight": 0.15, "raw": row_count_score, "contribution": round(0.15 * row_count_score, 4)},
        "age":             {"weight": 0.10, "raw": round(age_score, 4), "contribution": round(0.10 * age_score, 4)},
        "admission_date":  {"weight": 0.20, "raw": round(admission_score, 4), "contribution": round(0.20 * admission_score, 4)},
        "gender":          {"weight": 0.15, "raw": round(gender_score, 4), "contribution": round(0.15 * gender_score, 4)},
        "weight_kg":       {"weight": 0.10, "raw": round(weight_score, 4), "contribution": round(0.10 * weight_score, 4)},
        "blood_type":      {"weight": 0.15, "raw": round(blood_type_score, 4), "contribution": round(0.15 * blood_type_score, 4)},
        "readmitted":      {"weight": 0.15, "raw": round(readmitted_score, 4), "contribution": round(0.15 * readmitted_score, 4)},
    }
    for check, vals in breakdown.items():
        status = "✓" if vals["raw"] >= 0.999 else "✗"
        print(f"  {status} {check:<18} weight={vals['weight']:.2f}  "
              f"raw={vals['raw']:.4f}  contrib={vals['contribution']:.4f}")
    return


# ---------------------------------------------------------------------------
# __main__ — run dry-run sequence then diagnose
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from environment.env import DataCleaningEnv

    print("=" * 72)
    print("HARD TASK — Diagnostic Dry-Run")
    print("=" * 72)

    # Build the environment and obtain ground truth
    env = DataCleaningEnv("hard")
    grader_ref = HardGrader()
    ground_truth = grader_ref.truth_df.copy()

    # ── Apply the deterministic dry-run cleaning sequence ──────────────────
    # Step 1: drop_duplicates
    r = env.step({"action_type": "drop_duplicates"})
    print(f"[1] drop_duplicates        reward={r.reward:+.4f}  done={r.done}")

    # Step 2: fill_nulls(age, median)
    r = env.step({"action_type": "fill_nulls", "column": "age", "params": {"strategy": "median"}})
    print(f"[2] fill_nulls(age,median) reward={r.reward:+.4f}  done={r.done}")

    # Step 3: cast_column(age, int)
    r = env.step({"action_type": "cast_column", "column": "age", "params": {"dtype": "int"}})
    print(f"[3] cast(age,int)          reward={r.reward:+.4f}  done={r.done}")

    # Step 4: fill_nulls(weight_kg, median)
    r = env.step({"action_type": "fill_nulls", "column": "weight_kg", "params": {"strategy": "median"}})
    print(f"[4] fill_nulls(wt,median)  reward={r.reward:+.4f}  done={r.done}")

    # Step 5: cast_column(admission_date, datetime)
    r = env.step({"action_type": "cast_column", "column": "admission_date", "params": {"dtype": "datetime"}})
    print(f"[5] cast(admission,dt)     reward={r.reward:+.4f}  done={r.done}")

    # Step 6: remove_outliers(weight_kg, iqr)
    r = env.step({"action_type": "remove_outliers", "column": "weight_kg", "params": {"method": "iqr"}})
    print(f"[6] remove_outliers(wt)    reward={r.reward:+.4f}  done={r.done}")

    # Step 7: normalize_values(gender, mapping)
    r = env.step({
        "action_type": "normalize_values",
        "column": "gender",
        "params": {
            "mapping": {
                "M": "Male", "male": "Male", "1": "Male",
                "F": "Female", "female": "Female", "0": "Female",
            }
        },
    })
    print(f"[7] normalize(gender)      reward={r.reward:+.4f}  done={r.done}")

    # Step 8: normalize_values(blood_type, mapping)
    r = env.step({
        "action_type": "normalize_values",
        "column": "blood_type",
        "params": {
            "mapping": {
                "A_pos": "A+", "Apos": "A+", "a+": "A+",
                "A_neg": "A-", "Aneg": "A-", "a-": "A-",
                "B_pos": "B+", "Bpos": "B+", "b+": "B+",
                "B_neg": "B-", "Bneg": "B-", "b-": "B-",
                "AB_pos": "AB+", "ABpos": "AB+", "ab+": "AB+",
                "AB_neg": "AB-", "ABneg": "AB-", "ab-": "AB-",
                "O_pos": "O+", "Opos": "O+", "o+": "O+",
                "O_neg": "O-", "Oneg": "O-", "o-": "O-",
            }
        },
    })
    print(f"[8] normalize(blood_type)  reward={r.reward:+.4f}  done={r.done}")

    # Step 9: cast_column(readmitted, bool)
    r = env.step({"action_type": "cast_column", "column": "readmitted", "params": {"dtype": "bool"}})
    print(f"[9] cast(readmitted,bool)  reward={r.reward:+.4f}  done={r.done}")

    # ── Snapshot of cleaned df before submit ──────────────────────────────
    cleaned_df = env.current_df.copy()

    # Step 10: submit
    r = env.step({"action_type": "submit"})
    print(f"[10] submit                reward={r.reward:+.4f}  done={r.done}")
    print(f"\nGrader score at submit: {r.info.grader_score}")
    print(f"Episode total reward  : {env.episode_reward:.4f}")

    # ── Run diagnostic ────────────────────────────────────────────────────
    diagnose(cleaned_df, ground_truth)
