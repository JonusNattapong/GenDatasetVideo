from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import os
import json
from datetime import datetime
import logging
import aioredis
from celery.result import AsyncResult
from functools import wraps
from .celery_app import celery_app, generate_video_task

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

# --- Pydantic Models ---
class SVDParams(BaseModel):
    resolution: str = Field("1024x576", description="Video resolution e.g., '1024x576'")
    seed: int = Field(-1, description="Random seed, -1 for random")
    motion_bucket_id: int = Field(127, description="Motion bucket ID for SVD")
    fps: int = Field(6, description="Frames per second for the generated video")
    noise_aug_strength: float = Field(0.02, description="Noise augmentation strength for SVD")

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation")
    dataset_name: str = Field("default_dataset", description="Name of the dataset to add the video to")
    parameters: SVDParams = Field(default_factory=SVDParams, description="SVD generation parameters")

# --- Global Variables / State ---
redis: aioredis.Redis | None = None  # Redis connection for caching

# --- FastAPI App ---
app = FastAPI(title="GenDatasetVideo API")

@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection."""
    global redis
    redis = await aioredis.create_redis_pool('redis://localhost')
    logger.info("Connected to Redis cache")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources."""
    if redis is not None:
        redis.close()
        await redis.wait_closed()
        logger.info("Closed Redis connection")

def cache(expire=3600):
    """Cache decorator for API endpoints."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"cache:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached = await redis.get(key)
            if cached:
                return json.loads(cached)
            
            # If not in cache, execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await redis.set(key, json.dumps(result), expire=expire)
            
            return result
        return wrapper
    return decorator

# --- API Endpoints ---

@app.get("/api/health", tags=["Status"])
@cache(expire=30)  # Cache health check for 30 seconds
async def health_check():
    """Enhanced health check including Celery and Redis status."""
    # Check Redis
    redis_status = "ok" if redis is not None else "not connected"
    
    # Check Celery
    try:
        i = celery_app.control.inspect()
        celery_status = "ok" if i.ping() else "not responding"
    except Exception as e:
        celery_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "services": {
            "redis": redis_status,
            "celery": celery_status
        }
    }

@app.post("/api/generate", tags=["Generation"])
async def generate_video_endpoint(request: GenerateRequest):
    """Asynchronously generates video using Celery task."""
    
    # Create Celery task
    task = generate_video_task.delay(request.dict())
    
    return JSONResponse({
        "status": "processing",
        "task_id": task.id,
        "message": "Video generation started",
        "status_url": f"/api/tasks/{task.id}"
    })

@app.get("/api/tasks/{task_id}", tags=["Tasks"])
async def get_task_status(task_id: str):
    """Gets the status of a generation task."""
    task = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task.status,
    }
    
    if task.successful():
        response["result"] = task.result
    elif task.failed():
        response["error"] = str(task.result)
    elif task.status == 'PROGRESS':
        response["progress"] = task.info
        
    return response

# Rest of the endpoints (datasets, media serving, etc.) remain mostly unchanged
# Just add caching where appropriate

if __name__ == "__main__":
    logger.info(f"Frontend directory: {os.path.abspath(FRONTEND_DIR)}")
    logger.info(f"Datasets directory: {os.path.abspath(DATASETS_DIR)}")
    logger.info("Starting server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
