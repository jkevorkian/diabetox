"""
-----------------------------------------------------
MODULAR MEDICAL NOTE DATA EXTRACTION PIPELINE
-----------------------------------------------------
Supports:
  - LLM-based extraction (e.g., GPT, Gemini)
  - Regex-based extraction (fast, deterministic)
  - NLP-based extraction (placeholder for ML models)

You can modify:
  - The extraction method (via config["method"])
  - The instruction message
  - The output schema

-----------------------------------------------------
"""

import os
import re
import json
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod

# Optional dependencies
try:
    import openai
except ImportError:
    openai = None


# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------

CONFIG = {
    "input_file": "patients_data.json",
    "output_file": "patients_structured.csv",
    "method": "nlp",  # options: "local_llm", "llm", "regex", "nlp"

    # This prompt applies only to LLM extraction
    "instruction_prompt": """
    Extract structured medical data from the following clinical note.
    Return a valid JSON object ONLY (no commentary) with the following keys:

    {
      "age": int or null,
      "BMI": float or null,
      "HbA1c_percent": float or null,
      "random_glucose_mg_dL": float or null,
      "smoking_status": "current"|"past"|"never"|null,
      "history_hypertension": true|false|null,
      "history_heart_disease": true|false|null
    }

    If a value is not mentioned, use null.
    """,

    # Expected output columns
    "output_schema": [
        "patient_id",
        "has_diabetes",
        "age",
        "BMI",
        "HbA1c_percent",
        "random_glucose_mg_dL",
        "smoking_status",
        "history_hypertension",
        "history_heart_disease"
    ]
}


# ----------------------------------------------------
# ABSTRACT BASE CLASS
# ----------------------------------------------------

class BaseExtractor(ABC):
    """Abstract base class for any extraction method."""

    @abstractmethod
    def extract(self, text: str) -> dict:
        pass


# ----------------------------------------------------
# LLM EXTRACTOR
# ----------------------------------------------------

class LLMExtractor(BaseExtractor):
    """Uses an LLM (OpenAI, Gemini, etc.) for flexible data extraction."""

    def __init__(self, instruction_prompt: str, model_name: str = "gpt-5"):
        self.instruction_prompt = instruction_prompt
        self.model_name = model_name
        if openai is not None:
            openai.api_key = os.getenv("OPENAI_API_KEY")

    def extract(self, text: str) -> dict:
        if openai is None:
            raise ImportError("OpenAI library not available in this environment.")
        prompt = f"{self.instruction_prompt}\n\nMedical Note:\n{text}"

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise medical extractor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            output = response["choices"][0]["message"]["content"]
            return self._parse_output(output)
        except Exception as e:
            print(f"[LLMExtractor] Error: {e}")
            return self._empty_result()

    def _parse_output(self, response_text: str) -> dict:
        cleaned = (
            response_text.strip().strip("`").replace("json", "").replace("JSON", "")
        )
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = {}
        return self._standardize(data)

    def _standardize(self, data: dict) -> dict:
        return {
            "age": data.get("age"),
            "BMI": data.get("BMI"),
            "HbA1c_percent": data.get("HbA1c_percent"),
            "random_glucose_mg_dL": data.get("random_glucose_mg_dL"),
            "smoking_status": data.get("smoking_status"),
            "history_hypertension": data.get("history_hypertension"),
            "history_heart_disease": data.get("history_heart_disease"),
        }

    def _empty_result(self) -> dict:
        return {k: None for k in CONFIG["output_schema"] if k not in ["patient_id", "has_diabetes"]}


# ----------------------------------------------------
# REGEX EXTRACTOR
# ----------------------------------------------------

class RegexExtractor(BaseExtractor):
    """Fast pattern-based extraction using regular expressions."""

    def extract(self, text: str) -> dict:
        try:
            return {
                "age": self._extract_age(text),
                "BMI": self._extract_float(r"\bBMI(?: of)? ([0-9]+(?:\.[0-9]+)?)", text),
                "HbA1c_percent": self._extract_float(r"HbA1c(?: of)? ([0-9]+(?:\.[0-9]+)?)%", text),
                "random_glucose_mg_dL": self._extract_float(r"random glucose(?: of)? ([0-9]+)", text),
                "smoking_status": self._extract_smoking(text),
                "history_hypertension": self._extract_bool("hypertension", text),
                "history_heart_disease": self._extract_bool("heart disease", text),
            }
        except Exception:
            return {k: None for k in CONFIG["output_schema"] if k not in ["patient_id", "has_diabetes"]}

    def _extract_age(self, text):
        match = re.search(r"(\d{1,3})\s*(?:years?|yo|yrs?)", text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_float(self, pattern, text):
        match = re.search(pattern, text, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def _extract_smoking(self, text):
        if re.search(r"\b(current|smokes|smoker)\b", text, re.IGNORECASE):
            return "current"
        elif re.search(r"\b(former|past|quit)\b", text, re.IGNORECASE):
            return "past"
        elif re.search(r"\b(non[-\s]?smoker|never)\b", text, re.IGNORECASE):
            return "never"
        return None

    def _extract_bool(self, keyword, text):
        if re.search(rf"\bno {keyword}\b", text, re.IGNORECASE):
            return False
        elif re.search(keyword, text, re.IGNORECASE):
            return True
        return None


# ----------------------------------------------------
# NLP EXTRACTOR
# ----------------------------------------------------

class NLPExtractor(BaseExtractor):
    def __init__(self):
        import spacy
        from spacy.matcher import Matcher
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        # BMI pattern
        bmi_pattern = [
            {"LOWER": "bmi"},
            {"IS_ASCII": True, "OP": "*"},
            {"LIKE_NUM": True}
        ]
        self.matcher.add("BMI_PATTERN", [bmi_pattern])

        # HbA1c pattern
        hba1c_pattern = [
            {"LOWER": {"IN": ["hba1c", "hbA1c", "ha1c"]}},
            {"IS_ASCII": True, "OP": "*"},
            {"LIKE_NUM": True}
        ]
        self.matcher.add("HBA1C_PATTERN", [hba1c_pattern])

        # Glucose pattern
        gluc_pattern = [
            {"LOWER": {"IN": ["glucose", "glucose:"]}},
            {"IS_ASCII": True, "OP": "*"},
            {"LIKE_NUM": True}
        ]
        self.matcher.add("GLUCOSE_PATTERN", [gluc_pattern])

    def extract(self, text: str) -> dict:
        doc = self.nlp(text)
        matches = self.matcher(doc)
        result = {
            "age": None,
            "BMI": None,
            "HbA1c_percent": None,
            "random_glucose_mg_dL": None,
            "smoking_status": None,
            "history_hypertension": None,
            "history_heart_disease": None
        }
        # Age via regex fallback
        match = re.search(r"(\d{1,3})\s*(?:years?|yo|yrs?)", text, re.IGNORECASE)
        if match:
            result["age"] = int(match.group(1))

        for match_id, start, end in matches:
            match_name = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            # extract numeric
            for token in doc[end:end+3]:
                if token.like_num:
                    val = token.text
                    try:
                        if match_name == "BMI_PATTERN":
                            result["BMI"] = float(val)
                        elif match_name == "HBA1C_PATTERN":
                            result["HbA1c_percent"] = float(val)
                        elif match_name == "GLUCOSE_PATTERN":
                            result["random_glucose_mg_dL"] = float(val)
                    except:
                        pass
        # Smoking status
        if any(tok.lemma_.lower() == "smoke" for tok in doc):
            if any(tok.text.lower() in ["never", "non-smoker", "non smoker"] for tok in doc):
                result["smoking_status"] = "never"
            elif any(tok.lemma_.lower() in ["quit", "former", "past"] for tok in doc):
                result["smoking_status"] = "past"
            else:
                result["smoking_status"] = "current"
        # History hypertension
        if re.search(r"\bhistory of hypertension\b", text, flags=re.I):
            result["history_hypertension"] = True
        elif re.search(r"\bno history of hypertension\b", text, flags=re.I):
            result["history_hypertension"] = False
        # History heart disease
        if re.search(r"\bhistory of heart disease\b", text, flags=re.I):
            result["history_heart_disease"] = True
        elif re.search(r"\bno history of heart disease\b", text, flags=re.I):
            result["history_heart_disease"] = False

        return result



# ----------------------------------------------------
# LOCAL LLM EXTRACTOR
# ----------------------------------------------------


class LocalLLMExtractor(BaseExtractor):
    """
    Lightweight local LLM extractor using a Hugging Face model (e.g., Mistral, Gemma).
    Runs entirely in Colab/Kaggle without external API calls.
    """

    def __init__(self, instruction_prompt: str, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        from transformers import pipeline
        self.instruction_prompt = instruction_prompt
        self.model_name = model_name
        print(f"[LocalLLMExtractor] Loading model: {model_name} ...")
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            torch_dtype="auto",
            max_new_tokens=512
        )

    def extract(self, text: str) -> dict:
        """Generate structured JSON output from the note using the local model."""
        # Build the same prompt you used for GPT
        prompt = f"{self.instruction_prompt}\n\nMedical Note:\n{text}\n\nJSON Output:"
        try:
            output = self.pipe(prompt, do_sample=False)[0]["generated_text"]
        except Exception as e:
            print(f"[LocalLLMExtractor] Error: {e}")
            return self._empty_result()

        # Try to isolate JSON portion
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if not match:
            print("[LocalLLMExtractor] Warning: No JSON found in output.")
            return self._empty_result()

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            print("[LocalLLMExtractor] JSON parse failed.")
            data = {}

        return self._standardize(data)

    def _standardize(self, data: dict) -> dict:
        return {
            "age": data.get("age"),
            "BMI": data.get("BMI"),
            "HbA1c_percent": data.get("HbA1c_percent"),
            "random_glucose_mg_dL": data.get("random_glucose_mg_dL"),
            "smoking_status": data.get("smoking_status"),
            "history_hypertension": data.get("history_hypertension"),
            "history_heart_disease": data.get("history_heart_disease"),
        }

    def _empty_result(self) -> dict:
        return {k: None for k in CONFIG["output_schema"] if k not in ["patient_id", "has_diabetes"]}


# ----------------------------------------------------
# FACTORY FOR EXTRACTION METHODS
# ----------------------------------------------------

class ExtractorFactory:
    """Creates the correct extractor object based on config."""

    @staticmethod
    def create(method: str, instruction_prompt: str = None):
        if method == "llm":
            return LLMExtractor(instruction_prompt)
        elif method == "regex":
            return RegexExtractor()
        elif method == "nlp":
            return NLPExtractor()
        elif method == "local_llm":
            return LocalLLMExtractor(instruction_prompt)
        else:
            raise ValueError(f"Unknown extraction method: {method}")


# ----------------------------------------------------
# PIPELINE FUNCTIONS
# ----------------------------------------------------

def process_notes(df: pd.DataFrame, extractor: BaseExtractor) -> pd.DataFrame:
    """Runs the extraction for each record."""
    structured = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing notes"):
        note = row.get("medical_note", "")
        extracted = extractor.extract(note)
        extracted["patient_id"] = row.get("patient_id")
        extracted["has_diabetes"] = row.get("has_diabetes")
        structured.append(extracted)

    return pd.DataFrame(structured)


def save_to_csv(df: pd.DataFrame, output_path: str):
    """Saves DataFrame to CSV."""
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[+] Saved structured dataset to: {output_path}")


# ----------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------

def main():
    # Load dataset
    input_file = CONFIG["input_file"]
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")
    df = pd.read_json(input_file)
    print(f"[+] Loaded {len(df)} records")

    # Choose extraction method
    extractor = ExtractorFactory.create(CONFIG["method"], CONFIG.get("instruction_prompt"))

    # Process notes
    structured_df = process_notes(df, extractor)

    # Ensure consistent columns
    for col in CONFIG["output_schema"]:
        if col not in structured_df.columns:
            structured_df[col] = None
    structured_df = structured_df[CONFIG["output_schema"]]

    # Save output
    save_to_csv(structured_df, CONFIG["output_file"])
    print("[✓] Pipeline completed successfully.")


# ----------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------
if __name__ == "__main__":
    main()
