"""Debug script for hard task step-by-step analysis."""
from environment.env import DataCleaningEnv

env = DataCleaningEnv("hard")
print("Initial dirty_columns:", env.grader.dirty_columns(env.current_df))
print("Initial partial:", env.grader.partial_score(env.current_df))
print()

# Step 1: drop_duplicates
r = env.step({"action_type": "drop_duplicates"})
print(f"[1] drop_duplicates: reward={r.reward:+.4f}, applied={r.info.action_applied}")

# Step 2: fill_nulls(age, median)
r = env.step({"action_type": "fill_nulls", "column": "age", "params": {"strategy": "median"}})
print(f"[2] fill_nulls(age): reward={r.reward:+.4f}, applied={r.info.action_applied}")

# Step 3: cast_column(age, int)
r = env.step({"action_type": "cast_column", "column": "age", "params": {"dtype": "int"}})
print(f"[3] cast(age,int): reward={r.reward:+.4f}, applied={r.info.action_applied}")

# Step 4: fill_nulls(weight_kg, median)
r = env.step({"action_type": "fill_nulls", "column": "weight_kg", "params": {"strategy": "median"}})
print(f"[4] fill_nulls(wt): reward={r.reward:+.4f}, applied={r.info.action_applied}")

# Step 5: cast(admission_date, datetime)
r = env.step({"action_type": "cast_column", "column": "admission_date", "params": {"dtype": "datetime"}})
print(f"[5] cast(adm,dt): reward={r.reward:+.4f}, applied={r.info.action_applied}")

# Step 6: remove_outliers(weight_kg, iqr)
r = env.step({"action_type": "remove_outliers", "column": "weight_kg", "params": {"method": "iqr"}})
print(f"[6] remove_outliers(wt): reward={r.reward:+.4f}, applied={r.info.action_applied}")
print(f"    column_deltas: {r.info.column_deltas}")

# Step 7: normalize gender
r = env.step({
    "action_type": "normalize_values", "column": "gender",
    "params": {"mapping": {
        "M": "Male", "male": "Male", "1": "Male",
        "F": "Female", "female": "Female", "0": "Female",
    }}
})
print(f"[7] normalize(gender): reward={r.reward:+.4f}, applied={r.info.action_applied}")

# Step 8: normalize blood_type
r = env.step({
    "action_type": "normalize_values", "column": "blood_type",
    "params": {"mapping": {
        "A_pos": "A+", "Apos": "A+", "a+": "A+",
        "A_neg": "A-", "Aneg": "A-", "a-": "A-",
        "B_pos": "B+", "Bpos": "B+", "b+": "B+",
        "B_neg": "B-", "Bneg": "B-", "b-": "B-",
        "AB_pos": "AB+", "ABpos": "AB+", "ab+": "AB+",
        "AB_neg": "AB-", "ABneg": "AB-", "ab-": "AB-",
        "O_pos": "O+", "Opos": "O+", "o+": "O+",
        "O_neg": "O-", "Oneg": "O-", "o-": "O-",
    }}
})
print(f"[8] normalize(bt): reward={r.reward:+.4f}, applied={r.info.action_applied}")

# Step 9: cast readmitted bool
r = env.step({"action_type": "cast_column", "column": "readmitted", "params": {"dtype": "bool"}})
print(f"[9] cast(readmit,bool): reward={r.reward:+.4f}, applied={r.info.action_applied}")
print(f"    column_deltas: {r.info.column_deltas}")
readmit_col = env.current_df["readmitted"]
print(f"    readmitted dtype: {readmit_col.dtype}")
print(f"    dirty_cols remaining: {env.grader.dirty_columns(env.current_df)}")

# Check weight_kg status
import pandas as pd
wt = env.current_df["weight_kg"]
print(f"\n    weight_kg range: {wt.min():.1f} - {wt.max():.1f}")
print(f"    weight_kg nulls: {wt.isna().sum()}")

# Submit
r = env.step({"action_type": "submit"})
print(f"\n[10] submit: reward={r.reward:+.4f}, score={r.info.grader_score}")
print(f"     grader_breakdown: {r.info.grader_breakdown}")
