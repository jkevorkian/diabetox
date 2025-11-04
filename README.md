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

 *
 */