import re
import json
import pandas as pd
from pathlib import Path
from typing import Optional

# ---------- FILE PATHS ----------
DATA_PATH = Path("patients_data.json")   # input file
OUT_CSV = Path("extracted_patients.csv") # output file


# ---------- REGEX PATTERNS ----------
age_re = re.compile(r'(\b\d{1,3})[ -]?(?:year|yr)s?\b[- ]?old', re.I)
gender_re = re.compile(r'\b(male|female)\b', re.I)
bmi_re = re.compile(r'\bBMI(?:\s*(?:is|:|of))?\s*([\d]{1,3}(?:\.\d+)?)\b', re.I)
hba1c_num_re = re.compile(r'\bHbA1c(?:\s*(?:is|:|of))?\s*([\d]{1,3}(?:\.\d+)?)%?\b', re.I)
rg_num_re = re.compile(r'\brandom(?:\s*blood|\s*glucose)?\s*(?:reading|level|is|:|of)?\s*([\d]{1,3}(?:\.\d+)?)\s*(?:mg/dL|mg/dl)?', re.I)
rg_num_re_alt = re.compile(r'\b(?:glucose|random glucose)\s*(?:of|:)?\s*([\d]{1,3}(?:\.\d+)?)\b', re.I)

# ---------- PHRASE LISTS ----------
pos_hypertension = [r'\bhistory of hypertension\b', r'\bhas hypertension\b', r'\bhypertensive\b']
neg_hypertension = [r'\bno history of hypertension\b', r'\bdenies hypertension\b', r'\bno hypertension\b', r'\bwithout hypertension\b']

pos_heart = [r'\bhistory of heart disease\b', r'\bhas heart disease\b', r'\bcardiac disease\b', r'\bcardiovascular disease\b']
neg_heart = [r'\bno history of heart disease\b', r'\bno heart disease\b', r'\bdenies heart disease\b', r'\bwithout heart disease\b']

smoking_current = [r'\bcurrent smoker\b', r'\bcurrently smoking\b', r'\bsmokes\b', r'\bcurrent smoking\b']
smoking_past = [r'\bpast smoker\b', r'\bformer smoker\b', r'\bquit smoking\b', r'\bhistory of smoking\b']
smoking_never = [r'\bnon-smoker\b', r'\bdoes not smoke\b', r'\bnever smoked\b', r'\bno smoking history\b']

# textual categories for HbA1c / Glucose
hba1c_text = [(r'\bvery high hba1c\b',3),(r'\bhigh hba1c\b',2),(r'\belevated hba1c\b',2),(r'\bnormal hba1c\b',1),(r'\blow hba1c\b',0)]
rg_text = [(r'\bvery high random glucose\b',3),(r'\bhigh random glucose\b',2),(r'\belevated random glucose\b',2),(r'\bnormal random glucose\b',1),(r'\blow random glucose\b',0)]

negation_cues = re.compile(r'\b(no|denies|without|not)\b', re.I)

# ---------- HELPERS ----------
def is_negated(note: str, span_start: int):
    window = note[max(0, span_start-40):span_start]
    return bool(negation_cues.search(window))

def extract_age(note: str) -> Optional[int]:
    m = age_re.search(note)
    return int(m.group(1)) if m else None

def extract_gender(note: str) -> Optional[str]:
    m = gender_re.search(note)
    if m:
        return "Male" if m.group(1).lower().startswith("m") else "Female"
    return None

def extract_bmi(note: str) -> Optional[float]:
    m = bmi_re.search(note)
    return float(m.group(1)) if m else None

def extract_boolean(note: str, pos_patterns, neg_patterns) -> int:
    n = note.lower()
    for p in neg_patterns:
        if re.search(p, n):
            return 0
    for p in pos_patterns:
        m = re.search(p, n)
        if m and not is_negated(n, m.start()):
            return 1
    return 0

def extract_smoking(note: str) -> str:
    n = note.lower()
    for p in smoking_never:
        if re.search(p, n):
            return "never"
    for p in smoking_current:
        m = re.search(p, n)
        if m and not is_negated(n, m.start()):
            return "current"
    for p in smoking_past:
        m = re.search(p, n)
        if m and not is_negated(n, m.start()):
            return "past"
    return "not known"

def extract_hba1c(note: str) -> Optional[float]:
    m = hba1c_num_re.search(note)
    if m:
        return float(m.group(1))
    for pat, val in hba1c_text:
        if re.search(pat, note, re.I):
            return float(val)
    return None

def hba1c_to_ordinal(val: Optional[float]) -> Optional[int]:
    if val is None: return None
    if val < 4.5: return 0
    if 4.5 <= val <= 5.6: return 1
    if 5.7 <= val <= 6.4: return 2
    if val >= 6.5: return 3
    return None

def extract_random_glucose(note: str) -> Optional[float]:
    m = rg_num_re.search(note) or rg_num_re_alt.search(note)
    if m:
        return float(m.group(1))
    for pat, val in rg_text:
        if re.search(pat, note, re.I):
            return float(val)
    return None

def rg_to_ordinal(val: Optional[float]) -> Optional[int]:
    if val is None: return None
    if val < 80: return 0
    if 80 <= val <= 140: return 1
    if 141 <= val <= 199: return 2
    if val >= 200: return 3
    return None


# ---------- MAIN EXTRACTION ----------
def extract_all(records):
    rows = []
    for r in records:
        note = r.get("medical_note", "")
        rows.append({
            "patient_id": r.get("patient_id"),
            "Age": extract_age(note),
            "Gender": extract_gender(note),
            "Hypertension": extract_boolean(note, pos_hypertension, neg_hypertension),
            "Heart Disease": extract_boolean(note, pos_heart, neg_heart),
            "Smoking History": extract_smoking(note),
            "BMI": extract_bmi(note),
            "HbA1c_raw": extract_hba1c(note),
            "HbA1c": hba1c_to_ordinal(extract_hba1c(note)),
            "RandomGlucose_raw": extract_random_glucose(note),
            "Random Glucose": rg_to_ordinal(extract_random_glucose(note))
        })
    return pd.DataFrame(rows)


# ---------- RUN SCRIPT ----------
if __name__ == "__main__":
    print("Loading data...")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    df = extract_all(data)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Extraction completed. Saved to {OUT_CSV}\n")

    print("Missing value summary:")
    print(df.isna().sum())
    print("\nSmoking distribution:")
    print(df["Smoking History"].value_counts(dropna=False))
    print("\nPreview:")
    print(df.head(10))
