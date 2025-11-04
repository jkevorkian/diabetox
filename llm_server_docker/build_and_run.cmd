# Build
docker build -t local-llm-server .

# Run
docker run -d -p 8000:8000 -v "${PWD}/models:/app/models" local-llm-server

