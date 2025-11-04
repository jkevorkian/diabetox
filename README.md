## JSON interface (concise spec)

Input (patient JSON object)

patient_id (integer) — unique id for the patient (optional for ad-hoc notes).

has_diabetes (0 or 1) — ground truth label if known (optional).

medical_note (string) — the free-text clinical note to extract from.

Output (one record per patient, merged extraction + prediction)

patient_id: integer or null

Age: integer or null

Gender: "Male" | "Female" | "other" | null

Hypertension: 0 | 1 | null

Heart Disease: 0 | 1 | null

Smoking History: "never" | "past" | "current" | "not known" | null

BMI: float or null

HbA1c: "Low" | "Normal" | "High" | "Very High" | null

Random Glucose: "Low" | "Normal" | "High" | "Very High" | null

medical_note: original string

has_diabetes: original label if present or null

predicted_prob: number (0–100) — classifier output probability percent

The webpage code enforces this shape as the extraction target and uses a fixed feature mapping to feed the placeholder NN.

## How extraction works (and how to plug your LLMStudio)

The script first tries to call local LLMStudio endpoint http://localhost:1234/v1/chat/completions using a payload very similar to your Python code (same instructions JSON).

If the endpoint call fails (CORS or not running), the page falls back to the rule-based extractor implemented in-browser (regex + heuristics). This ensures the UI is functional without a server LLM.

If you run LLMStudio locally and it's accessible from the browser, the page will parse the JSON returned by the model (it attempts to extract the JSON object inside the model's text).

If you prefer not to call the LLM from the browser: toggle useLLM = false; in the JS to disable LLM calls and always use rule-based extraction.

## Neural network placeholder — notes for replacing with a trained model

The placeholder NN is a tiny feed-forward network implemented in JS with:

inputSize = 8 features (Age, Gender, Hypertension, Heart Disease, Smoking, BMI, HbA1c_score, RandomGlucose_score)

one hidden layer (12 units), ReLU, sigmoid output → probability.

Default weights are randomized. To use trained weights:

Replace nnWeights with your serialized trained weights (same shape: W1 is inputSize x hiddenSize, b1 length hiddenSize, W2 hiddenSize x 1, b2 length 1).

Optionally implement a training UI or an import button to load JSON weights.

If you plan to run training in-browser using TF.js, you can convert the above mapping to a tf model and load weights — the UI already exposes viewModelBtn which prints a preview; extend that to import weights.

## How to use the page

Paste a single medical note into the textbox and click Extract & Predict (it will try the LLM, fallback to rule-based if unavailable).

Or click Load file after selecting a patients.json file that is an array of patient objects in the input format; or click the load demo dataset if no file selected.

Or enter an ID and click Fetch to call https://api.hackupm2025.workers.dev/api/v1/patients/train/{id}; the response shape can be either a single object or array — the page will handle both.

Results show a per-patient card (first 5), a table with the extracted columns and predicted_prob and a JSON preview of the last extracted record. Use Export CSV to download results.

## Implementation caveats & suggestions

CORS: If you call your local LLMStudio API from the browser, ensure the server allows cross-origin requests from file:// or where you serve the page, otherwise the browser will block the request. If CORS is an issue, hosting a tiny proxy server that your static page can call (or running the page from a simple local server like python -m http.server) will help.

Security: The embedded demo uses random weights; do not ship randomized or unvalidated clinical predictions into production.

Model training: For realistic predictions you will want to:

Collect extracted columns for many patients (CSV export helps).

Train a proper model (server-side or in-browser with TF.js).

Replace placeholder weights with produced model weights and/or implement an upload/replace weights UI.

Extractor reliability: The in-browser rule-based extractor is intentionally conservative. When using a proper LLM extractor, prefer temperature: 0.1 and a strict system instruction so the LLM returns valid JSON. Add server-side validation of the LLM output in production.
