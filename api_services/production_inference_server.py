"""
Instead of running inference inside your FastAPI process, you run a dedicated inference server: vLLM, Text Generation 
Inference (TGI), or NVIDIA Triton, and have FastAPI make async API calls to it

"""

import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI()

VLLM_URL = "http://localhost:8000/generate"  # Adjust if your vLLM runs elsewhere


@app.post("/generate")
async def generate(prompt: str) -> dict[str, str]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                VLLM_URL,
                json={"text": prompt, "max_tokens": 64},
                timeout=30.0,
            )
            response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  