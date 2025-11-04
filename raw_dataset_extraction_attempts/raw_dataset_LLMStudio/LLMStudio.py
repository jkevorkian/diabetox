import requests
import json
import pandas as pd

# Local LM Studio endpoint
API_URL = "http://localhost:1234/v1/chat/completions"

# Extraction prompt
instructions = """You are a clinical data extractor.
Given a patient description, return ONLY a JSON with these fields:

{
 "Age": integer,
 "Gender": "Male" or "Female",
 "Hypertension": 0 or 1,
 "Heart Disease": 0 or 1,
 "Smoking History": "never" | "past" | "current" | "not known",
 "BMI": float,
 "HbA1c": "Low" | "Normal" | "High" | "Very High",
 "Random Glucose": "Low" | "Normal" | "High" | "Very High"
}

Do not add explanations or text outside the JSON.
"""

# Model name (update if your local config differs)
MODEL_NAME = "qwen/qwen3-4b-2507"#"hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0"

# Load patient datas
with open("patients_data.json", "r", encoding="utf-8") as f:
    patients = json.load(f)

results = []

for i, patient in enumerate(patients, start=1):
    medtext = patient["medical_note"]
    has_diabetes = patient["has_diabetes"]

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": medtext}
        ],
        "temperature": 0.1
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"].strip()

        # Try to parse JSON output
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ Could not parse JSON for patient {i}, saving empty fields.")
            data = {}

        data["index"] = i
        data["has_diabetes"] = has_diabetes
        data["medical_note"] = medtext

        results.append(data)

    except Exception as e:
        print(f"❌ Error processing patient {i}: {e}")
        results.append({
            "index": i,
            "has_diabetes": has_diabetes,
            "medical_note": medtext
        })

# Convert to DataFrame and save
df = pd.DataFrame(results)
df.to_csv("extracted_patients_data.csv", index=False, encoding="utf-8")

print("✅ Extraction complete! Saved to extracted_patients_data.csv")
