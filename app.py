from flask import Flask, request, jsonify
import face_recognition
import cv2
import os
import numpy as np
import sqlite3
import threading

app = Flask(__name__)
DB_NAME = 'faces.db'

def create_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Create media table with processed status
    c.execute('''CREATE TABLE IF NOT EXISTS media (
                    id INTEGER PRIMARY KEY,
                    filename TEXT,
                    path TEXT,
                    processed BOOLEAN DEFAULT 0
                )''')
    
    # Create embeddings table
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY,
                    encoding BLOB,
                    media_id INTEGER,
                    FOREIGN KEY(media_id) REFERENCES media(id)
                )''')
    
    conn.commit()
    conn.close()

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    media_ids = []

    for file in files:
        media_id = save_media(file)
        media_ids.append(media_id)
        # Start a background thread for processing
        threading.Thread(target=process_media, args=(file, media_id)).start()

    return jsonify({"status": "success", "media_ids": media_ids})

def save_media(file):
    # Save the media file
    media_path = os.path.join('static', file.filename)
    file.save(media_path)

    # Insert media info into the database
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO media (filename, path) VALUES (?, ?)', (file.filename, media_path))
    media_id = c.lastrowid
    conn.commit()
    conn.close()

    return media_id

def process_media(file, media_id):
    if file.filename.endswith('.mp4'):
        process_video(file, media_id)
    else:
        process_image(file, media_id)
    
    mark_media_processed(media_id)

def mark_media_processed(media_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('UPDATE media SET processed = 1 WHERE id = ?', (media_id,))
    conn.commit()
    conn.close()

def process_image(file, media_id):
    image = face_recognition.load_image_file(file)
    face_encodings = face_recognition.face_encodings(image)
    unique_encodings = set()

    for encoding in face_encodings:
        if not is_duplicate(encoding):
            unique_encodings.add(encoding.tobytes())
    
    save_embeddings(unique_encodings, media_id)

def process_video(file, media_id, skip_frames=5):
    video_path = os.path.join('static', file.filename)
    file.save(video_path)
    
    video_capture = cv2.VideoCapture(video_path)
    unique_encodings = set()
    
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        if frame_count % skip_frames == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame)
            for encoding in face_encodings:
                if not is_duplicate(encoding):
                    unique_encodings.add(encoding.tobytes())
        
        frame_count += 1

    video_capture.release()
    os.remove(video_path)

    save_embeddings(unique_encodings, media_id)

def is_duplicate(encoding, threshold=0.6):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Fetch existing embeddings to check for similarity
    c.execute('SELECT encoding FROM embeddings')
    existing_encodings = c.fetchall()
    
    for existing_encoding in existing_encodings:
        existing_encoding = np.frombuffer(existing_encoding[0], dtype=np.float64)
        similarity = np.dot(encoding, existing_encoding) / (np.linalg.norm(encoding) * np.linalg.norm(existing_encoding))
        if similarity >= threshold:  # Check if the similarity is above the threshold
            conn.close()
            return True
    
    conn.close()
    return False

def save_embeddings(encodings, media_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    for encoding in encodings:
        c.execute('INSERT INTO embeddings (encoding, media_id) VALUES (?, ?)', (encoding, media_id))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_db()
    app.run(debug=True)
