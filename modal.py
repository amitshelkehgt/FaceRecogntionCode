import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from pymongo import MongoClient
from datetime import datetime, time
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("db_url"))
db = client["Face_Recognitions"]

cap = cv2.VideoCapture(os.getenv("rtsp_url"))
# cap = cv2.VideoCapture(0)  # Use 0 for webcam

modal = FaceAnalysis(providers=["CPUExecutionProvider"])
modal.prepare(ctx_id=0, det_size=(640, 640))

tolerance = 1.0


def run():
    while True:
        face_embeddings = list(db["face_embeddings"].find({}))
        known_face_embeddings = [np.array(i["embedding"]) for i in face_embeddings]

        success, frame = cap.read()

        if not success:
            print("Failed to capture frame")
            break

        if not known_face_embeddings:
            continue

        frame = cv2.resize(frame, (640, 640))
        faces = modal.get(frame)

        for face in faces:
            process_face(face, frame, face_embeddings, known_face_embeddings)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def process_face(face, frame, face_embeddings, known_face_embeddings):
    box = face.bbox.astype(int)
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    face_embedding = face.embedding / np.linalg.norm(face.embedding)

    face_distances = np.linalg.norm(
        np.array(known_face_embeddings) - face_embedding, axis=1
    )
    best_match_index = np.argmin(face_distances)

    if face_distances[best_match_index] > tolerance:
        return

    user = face_embeddings[best_match_index]

    user_id = ObjectId(user["_id"])
    name = user["name"]

    cv2.putText(
        frame,
        name,
        (box[0] + 6, box[3] - 6),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (255, 255, 255),
        1,
    )

    today = datetime.today()
    is_today_exist = db["Emp_Images"].find_one(
        {
            "user_id": user_id,
            "arrival_time": {
                "$gte": datetime.combine(today, time.min),
                "$lte": datetime.combine(today, time.max),
            },
        }
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    if is_today_exist:
        path = os.path.join("images", name, current_date, "departure.jpg")
        db["Emp_Images"].update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "departure_time": datetime.now(),
                    "departure_picture": path,
                }
            },
        )
    else:
        path = os.path.join("images", name, current_date, "arrival.jpg")
        db["Emp_Images"].insert_one(
            {
                "user_id": user_id,
                "arrival_time": datetime.now(),
                "arrival_picture": path,
            }
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.array(frame))


def get_embeddings(image: bytes):
    np_array = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return modal.get(image)


if __name__ == "__main__":
    run()
