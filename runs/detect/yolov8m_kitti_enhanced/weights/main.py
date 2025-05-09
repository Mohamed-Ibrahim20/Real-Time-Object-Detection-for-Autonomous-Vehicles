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

# --- Patched static and templates paths for App Service --- 
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"

if not os.path.exists(STATIC_DIR):
    print(f"WARNING: Static directory '{STATIC_DIR}' not found at startup in {os.getcwd()}")
if not os.path.exists(TEMPLATES_DIR):
    print(f"WARNING: Templates directory '{TEMPLATES_DIR}' not found at startup in {os.getcwd()}")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)
# --- End Patched Paths --- 

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS_IMG = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_VID = {'mp4', 'avi', 'mov'}
MODEL_PATH = "best.pt"  # This should now be in the root /home/site/wwwroot

if not os.path.exists(UPLOAD_FOLDER):
    print(f"Creating UPLOAD_FOLDER at {os.path.join(os.getcwd(), UPLOAD_FOLDER)}")
    os.makedirs(UPLOAD_FOLDER)

app.state.model = None

def load_model():
    abs_model_path = os.path.abspath(MODEL_PATH)
    print(f"Attempting to load YOLO model from: {abs_model_path}")
    if not os.path.exists(abs_model_path):
        print(f"CRITICAL ERROR: Model file NOT FOUND at {abs_model_path}")
        app.state.model = None
        app.state.names = {}
        return
    try:
        app.state.model = YOLO(MODEL_PATH) # Use relative MODEL_PATH as YOLO might handle it internally
        app.state.names = app.state.model.names
        print(f"Successfully loaded YOLO model from {abs_model_path}")
    except Exception as e:
        print(f"CRITICAL ERROR loading YOLO model: {e}")
        import traceback
        print(traceback.format_exc())
        app.state.model = None
        app.state.names = {}

@app.on_event("startup")
async def startup_event():
    print("FastAPI startup_event triggered.")
    load_model()
    print("FastAPI startup_event finished.")

@app.get("/health")
async def health_check():
    model_status = "loaded" if app.state.model is not None else "not loaded"
    return {"status": "healthy", "model_status": model_status}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def allowed_file(filename: str, extensions: set) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    img_resized = cv2.resize(img_bgr, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    return np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if app.state.model is None:
        print("Predict endpoint: Model not loaded!")
        raise HTTPException(status_code=500, detail="Model not loaded or failed to load. Check logs.")

    filename = file.filename
    if not allowed_file(filename, ALLOWED_EXTENSIONS_IMG.union(ALLOWED_EXTENSIONS_VID)):
        raise HTTPException(status_code=400, detail="File type not allowed")

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        with open(filepath, "wb") as f:
            f.write(await file.read())

        if allowed_file(filename, ALLOWED_EXTENSIONS_IMG):
            img = cv2.imread(filepath)
            if img is None:
                raise HTTPException(status_code=400, detail="Cannot read image file")
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
            # os.remove(filepath) # Keep file for now for debugging if needed
            return Response(content=buf.tobytes(), media_type='image/jpeg')
        else:
            raise HTTPException(status_code=400, detail="Video detection available via /upload-video and /stream/{filename}")
    except Exception as e:
        print(f"Error during /predict: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    finally:
        if os.path.exists(filepath):
             print(f"Removing uploaded file: {filepath}")
             os.remove(filepath)


# Re-introduce real-time video streaming endpoints
@app.post("/upload-video")
async def upload_video(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if app.state.model is None:
        print("Upload-video endpoint: Model not loaded!")
        raise HTTPException(status_code=500, detail="Model not loaded or failed to load. Check logs.")
        
    if not allowed_file(file.filename, ALLOWED_EXTENSIONS_VID):
        raise HTTPException(status_code=400, detail="Video file type not allowed")
        
    timestamp = int(time.time())
    unique_filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    try:
        with open(filepath, "wb") as f:
            f.write(await file.read())
        print(f"Video saved to {filepath}")
            
        stream_url = f"{request.url.scheme}://{request.url.netloc}/stream/{unique_filename}"
        
        return {
            "message": "Video uploaded successfully", 
            "filename": unique_filename, # Use the unique filename
            "stream_url": stream_url
        }
    except Exception as e:
        print(f"Error during /upload-video: {e}")
        import traceback
        print(traceback.format_exc())
        # Clean up file if upload fails mid-way
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Error uploading video: {str(e)}")

@app.get("/stream/{filename}")
def stream_video(filename: str):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    print(f"Streaming video from {filepath}")
    if not os.path.exists(filepath):
        print(f"Video file not found for streaming: {filepath}")
        raise HTTPException(status_code=404, detail="Video not found")
    
    if app.state.model is None:
        print("Stream endpoint: Model not loaded!")
        # Do not remove the file if model isn't loaded, as it might be a temporary issue
        raise HTTPException(status_code=500, detail="Model not loaded or failed to load. Check logs.")
        
    def generate():
        cap = cv2.VideoCapture(filepath)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video stream or cannot read frame from {filename}")
                    break
                    
                results = app.state.model(frame, imgsz=640)[0]
                
                for box, score, label in zip(
                    results.boxes.xyxy.cpu().numpy(),
                    results.boxes.conf.cpu().numpy(),
                    results.boxes.cls.cpu().numpy().astype(int)
                ):
                    if score < 0.5: 
                        continue
                        
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0) 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label_text = f"{app.state.names[label]} {score:.2f}"
                    (w, h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - h - baseline), (x1 + w, y1), color, cv2.FILLED)
                    cv2.putText(frame, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                ret2, jpeg = cv2.imencode('.jpg', frame)
                if not ret2:
                    print(f"Failed to encode frame to JPEG for {filename}")
                    continue
                    
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error during video frame generation for {filename}: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            print(f"Releasing video capture for {filename}")
            cap.release()
            if os.path.exists(filepath):
                print(f"Removing video file after streaming: {filepath}")
                os.remove(filepath)
                
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')


# Add entrypoint for running with Uvicorn (for local testing, not used by Gunicorn in App Service)
if __name__ == "__main__":
    import uvicorn
    print("Running Uvicorn locally...")
    load_model() # Load model for local Uvicorn run as well
    uvicorn.run(app, host="0.0.0.0", port=8000) 