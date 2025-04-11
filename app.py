from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from modal import get_embeddings
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os
import requests

# ===== MongoDB Setup =====
client = MongoClient(os.getenv("db_url"))
db = client["Face_Recognitions"]
face_embeddings = db["face_embeddings"]
video_sources = db["video_sources"]
callbacks = db["callbacks"]
callback_logs = db["callback_logs"]

app = FastAPI()

# ===== 1. Register Face =====
@app.post("/register")
def register(
    name: str,
    image: UploadFile = File(...),
):
    image_bytes = image.file.read()
    faces = get_embeddings(image_bytes)

    if len(faces) == 0:
        return PlainTextResponse(status_code=404, content="No face found")
    if len(faces) > 1:
        return PlainTextResponse(status_code=400, content="Multiple faces found")

    embedding = faces[0].embedding / np.linalg.norm(faces[0].embedding)
    id = face_embeddings.insert_one({"name": name, "embedding": embedding.tolist()}).inserted_id

    return PlainTextResponse(content=f"Face registered successfully: {id}", status_code=200)

# ===== 2. Register RTSP Video Source =====
class VideoSource(BaseModel):
    source_name: str
    rtsp_url: str

@app.post("/register_video_source")
def register_video_source(source: VideoSource):
    video_sources.update_one(
        {"source_name": source.source_name},
        {"$set": {"rtsp_url": source.rtsp_url}},
        upsert=True
    )
    return {"message": f"Video source '{source.source_name}' registered successfully."}

# ===== 3. Register Callback/Webhook with Event =====
class CallbackRequest(BaseModel):
    name: str
    callback_url: str
    event: str  # e.g., "sign-in", "sign-out"

@app.post("/register_callback")
def register_callback(cb: CallbackRequest):
    callbacks.update_one(
        {"name": cb.name},
        {"$set": {
            "url": cb.callback_url,
            "event": cb.event
        }},
        upsert=True
    )
    return {"message": f"Callback for '{cb.name}' registered with event '{cb.event}'."}

# ===== 4. Trigger Callback (Simulate Detection) =====
@app.post("/trigger_callback")
def trigger_callback(name: str):
    callback = callbacks.find_one({"name": name})
    if not callback:
        return PlainTextResponse(content="Callback not found.", status_code=404)

    payload = {
        "event": callback.get("event", "unknown"),
        "name": name,
        "confidence": 0.98,
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        response = requests.post(callback["url"], json=payload)
        callback_logs.insert_one({
            "name": name,
            "url": callback["url"],
            "payload": payload,
            "status": response.status_code,
            "timestamp": datetime.utcnow()
        })
        return {"status": response.status_code, "message": "Callback triggered."}
    except Exception as e:
        return PlainTextResponse(content=str(e), status_code=500)

# ===== 5. View Callback Logs =====
@app.get("/callback_logs")
def get_callback_logs():
    logs = list(callback_logs.find({}, {"_id": 0}))
    return logs
