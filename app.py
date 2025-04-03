from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
from modal import get_embeddings
import numpy as np
from pymongo import MongoClient
import os


client = MongoClient(os.getenv("db_url"))
db = client["Face_Recognitions"]
face_embeddings = db["face_embeddings"]


app = FastAPI()


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
    face_embeddings.insert_one({"name": name, "embedding": embedding.tolist()})

    return PlainTextResponse(content="Face registered successfully", status_code=200)
