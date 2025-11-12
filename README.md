/**
 * # Project Documentation
 *
 * ## Overview
 * This proyect aids in diabetes detection using AI
 *
 * ## Deployment
 * you need docker installed on the device on which to run the proyect and it should be on windows
 * run Build docker build -t local-llm-server .
 * if successful, run docker run -d -p 8000:8000 -v "${PWD}/models:/app/models" local-llm-server
 * execute notebook cells on diabetox.ipynb in order
 * Then execute data_cleaning_pre_training, Then training, Then data_cleaning_post_training
 *
 * ## Data Retrieval Process
 * - The application initiates a request to the specified API endpoint and savess iterating along the API response limits.
 * - Upon receiving the full dataset, it is all saved on a single json file.
 * - This data is then transformed using an LLM from a medical description of the patient to a csv file with the following columns we will use as features:
    -
    -
    -
    -


 * ## Decisions Taken
 * - the values for HbA1c and Random Glucose extracted from the medical notes will, when not explicitly expressed as "high/low/medium" values instead of numeric ones, be put in said brackets

* ## Key Points
* - Dockerized local LLM to be deployed on any sistem allows for ON PREMISE OFFLINE EXECUTION OF THE SOLUTION given a middle range computing power hardware.
* - Tried implementing various AI models to check performance metrics and compare before going with LightGBM (Random Forest, various Neural networks, Linear regression, XGMBoost)
* - Implemented web interface MVP that allows the imput of patients diagnostics of various sources for extra flexibility and implementation use cases (thinking of a real-world use for this first approach):
   -Direct textbox medical text input for real-time diagnosis
   -Json standard format upload for diagnosing various patients in series (local database use case)
   -API retrieval of either one or more patient via their ID (assuming a medical DB or a medical doctor would have this information). Current usage of this is limited since the API endpoint used for the Hackathon is no longer available, but the code is flexible to maintain it with a new endpoint.
 */