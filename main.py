import os
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, Response, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import time

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS_IMG = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_VID = {'mp4', 'avi', 'mov'}
MODEL_PATH = "best.pt"  # Ensure best.pt is in /app

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.state.model = None


def load_model():
    try:
        app.state.model = YOLO(MODEL_PATH)
        # Store class names mapping
        app.state.names = app.state.model.names
        print(f"PyTorch model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        app.state.model = None


def allowed_file(filename: str, extensions: set) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    img_resized = cv2.resize(img_bgr, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    return np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]


def run_inference(input_tensor: np.ndarray):
    sess = app.state.ort_session
    if sess is None:
        return None, None, None
    inputs = {sess.get_inputs()[0].name: input_tensor}
    output_names = [o.name for o in sess.get_outputs()]
    raw = sess.run(output_names, inputs)
    # Assuming model outputs three arrays: boxes, scores, labels
    if len(raw) >= 3:
        return raw[0], raw[1], raw[2]
    else:
        return None, None, None


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": app.state.model is not None}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    filename = file.filename
    if not allowed_file(filename, ALLOWED_EXTENSIONS_IMG.union(ALLOWED_EXTENSIONS_VID)):
        raise HTTPException(status_code=400, detail="File type not allowed")

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        with open(filepath, "wb") as f:
            f.write(await file.read())

        # Perform inference, draw boxes, and return annotated media
        if allowed_file(filename, ALLOWED_EXTENSIONS_IMG):
            img = cv2.imread(filepath)
            if img is None:
                raise HTTPException(status_code=400, detail="Cannot read image file")
            # Inference at original resolution for better accuracy
            results = app.state.model(img, imgsz=640)[0]
            for box, score, label in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy().astype(int)
            ):
                x1, y1, x2, y2 = map(int, box)
                color = (255, 0, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label_text = f"{app.state.names[label]} {score:.2f}"
                (w, h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - h - baseline), (x1 + w, y1), color, cv2.FILLED)
                cv2.putText(img, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            _, buf = cv2.imencode('.jpg', img)
            os.remove(filepath)
            return Response(content=buf.tobytes(), media_type='image/jpeg')
        else:
            # For video, use the streaming endpoints
            raise HTTPException(status_code=400, detail="Video detection available via /upload-video and /stream/{filename}")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# Re-introduce real-time video streaming endpoints
@app.post("/upload-video")
async def upload_video(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    if not allowed_file(file.filename, ALLOWED_EXTENSIONS_VID):
        raise HTTPException(status_code=400, detail="Video file type not allowed")
        
    # Generate a unique filename to avoid collisions
    timestamp = int(time.time())
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save the uploaded file
    with open(filepath, "wb") as f:
        f.write(await file.read())
        
    # Construct streaming URL
    stream_url = f"{request.url.scheme}://{request.url.netloc}/stream/{filename}"
    
    return {
        "message": "Video uploaded successfully", 
        "filename": filename,
        "stream_url": stream_url
    }

@app.get("/stream/{filename}")
def stream_video(filename: str):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Video not found")
    
    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    def generate():
        cap = cv2.VideoCapture(filepath)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Run model inference on the frame
                results = app.state.model(frame, imgsz=640)[0]
                
                # Draw bounding boxes
                for box, score, label in zip(
                    results.boxes.xyxy.cpu().numpy(),
                    results.boxes.conf.cpu().numpy(),
                    results.boxes.cls.cpu().numpy().astype(int)
                ):
                    if score < 0.5:  # Apply confidence threshold
                        continue
                        
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0)  # Green color for better visibility
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add text label
                    label_text = f"{app.state.names[label]} {score:.2f}"
                    (w, h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - h - baseline), (x1 + w, y1), color, cv2.FILLED)
                    cv2.putText(frame, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Convert frame to JPEG for streaming
                ret2, jpeg = cv2.imencode('.jpg', frame)
                if not ret2:
                    continue
                    
                # Yield frame for streaming response
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')


# Add entrypoint for running with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000) 