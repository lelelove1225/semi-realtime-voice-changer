from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio
import aivis_speech


class Query(BaseModel):
    text: str


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/synthesis")
async def synthesis(query: Query, background_tasks: BackgroundTasks):
    text = query.text
    print(f"Received text: {text}")
    background_tasks.add_task(
        asyncio.run, aivis_speech.get_wav_and_play(text, "888753762")
    )
    return {"status": "ok"}
