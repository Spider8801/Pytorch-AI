"""
A ProcessPoolExecutor achieves this. The executor runs each inference call in a separate process, and the event loop can 
continue handling other requests while the process pool works

One important note on RAM: each worker process loads its own copy of the model. Two workers with a 2GB model means 4GB 
consumed before a single request arrives. Size your worker pool with this in mind.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI
from contextlib import asynccontextmanager
from transformers import pipeline


executor = ProcessPoolExecutor(max_workers=4)

def run_inference(prompt: str) -> str:
    # each process runs an isolated process, no GIL required

    pipe = pipeline("text-generation", model = "mistral/Mistral-7B-v0.1", device= 0)
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    output = pipe(prompt, max_new_tokens=64)

    return output[0]["generated_text"].split(prompt)[1]

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def generate(prompt: str) -> dict[str, str]:
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, run_inference, prompt)
    return {"text": result} 