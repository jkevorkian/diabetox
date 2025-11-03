import requests

BASE_URL = "https://api.hackupm2025.workers.dev/api/v1/patients/train"
LIMIT = 100

def fetch_all_patients():
    page = 1
    all_patients = []

    while True:
        params = {"page": page, "limit": LIMIT}
        response = requests.get(BASE_URL, params=params)

        # Raise for bad responses (4xx, 5xx)
        response.raise_for_status()

        data = response.json()

        # Extract the patients from the 'data' field
        patients = data.get("data", [])
        all_patients.extend(patients)

        # Extract pagination info
        pagination = data.get("pagination", {})
        has_next = pagination.get("hasNextPage", False)

        print(f"Fetched page {page} → {len(patients)} records")

        if not has_next:
            break  # Stop when no more pages
        page += 1

    print(f"\n✅ Done! Total patients fetched: {len(all_patients)}")
    return all_patients


if __name__ == "__main__":
    all_data = fetch_all_patients()

    # Example: write results to a JSON file
    import json
    with open("patients_data.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("💾 Saved all patients to patients_data.json")
