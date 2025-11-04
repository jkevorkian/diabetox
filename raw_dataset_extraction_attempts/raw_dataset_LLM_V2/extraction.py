from transformers import pipeline
import json
import csv

# Load a small, free Hugging Face model
# (TinyLlama is CPU-friendly and doesn't need a token)
generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=300,
)

def extract_features(note: str):
    system_prompt = """You are a clinical data extractor.
Given a patient description, return a JSON object with these fields:

{
 "Age": integer,
 "Gender": "Male" or "Female",
 "Hypertension": 0 or 1,
 "Heart Disease": 0 or 1,
 "Smoking History": "never" | "past" | "current" | "not known",
 "BMI": float,
 "HbA1c": integer (Low=0, Normal=1, High=2, Very High=3),
 "Random Glucose": integer (Low=0, Normal=1, High=2, Very High=3)
}

Do not add explanations or text outside the JSON."""

    prompt = f"{system_prompt}\n\nPatient note:\n{note}\n\nJSON:"
    result = generator(prompt, temperature=0.2)[0]["generated_text"]

    # Extract the JSON portion
    try:
        json_part = result.split("{", 1)[1]
        json_part = "{" + json_part
        json_part = json_part.split("}", 1)[0] + "}"
        data = json.loads(json_part)
    except Exception:
        data = {"Gender": None, "Age": None, "Hypertension": None,
                "Heart_Disease": None, "Smoking_History": None,
                "BMI": None, "HbA1c": None, "Random_Glucose": None}

    return data


def main():
    INPUT_FILE = "patients_data.json"
    OUTPUT_FILE = "patients_structured.csv"

    with open(INPUT_FILE, "r") as f:
        patients = json.load(f)

    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        fieldnames = [
            "patient_id", "Gender", "Age", "Hypertension", "Heart_Disease",
            "Smoking_History", "BMI", "HbA1c", "Random_Glucose"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for p in patients:
            features = extract_features(p["medical_note"])
            features["patient_id"] = p["patient_id"]
            writer.writerow(features)

    print(f"✅ Saved extracted data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
