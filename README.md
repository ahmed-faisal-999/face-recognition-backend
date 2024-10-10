# Face Recognition Directory

## Overview

The **Face Recognition Directory** is a tool designed to recognize and store unique face embeddings from uploaded images and videos. The project consists of two components: a Python backend that handles the face recognition processing and a React frontend that provides an interface for file uploads.

### Features
- Asynchronously processes images and videos to recognize faces.
- Stores media file details and unique face embeddings in a SQLite database.
- Checks for duplicate face embeddings using cosine similarity.
- Provides a user-friendly interface for uploading files.

---

## Backend (Python Flask)

### Description
The backend is built using Flask and leverages the `face_recognition` library for processing images and videos. It stores media and face embedding data in a SQLite database.

### How It Works
1. **File Upload:** Users can upload images or videos through the frontend.
2. **Media Processing:** Each uploaded file is processed in the background to extract face embeddings.
3. **Database Storage:** Media file details and unique face embeddings are stored in a SQLite database.

### Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd face-recognition-backend

2. **Install Dependencies: Create a virtual environment (optional but recommended) and install the required packages:**

    ```bash
    pip install -r requirements.txt

3. **Run the Backend:**

    ```bash
    python app.py