from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/llama-3.2-1b-instruct-q8_0.gguf")

app = FastAPI(title="Local LLM Server")

print(f"🔧 Loading model from {MODEL_PATH} ...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=4,
    n_gpu_layers=-1
)
print("✅ Model loaded successfully")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    output = llm.create_chat_completion(
        messages=[{"role": m.role, "content": m.content} for m in req.messages],
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )
    return output
