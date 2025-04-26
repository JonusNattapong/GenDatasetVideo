from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import os
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image
import uuid
import json
from datetime import datetime
import asyncio
import logging
import aioredis
from celery.result import AsyncResult
from functools import wraps
from .celery_app import celery_app, generate_video_task

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Assume the frontend files are in ../frontend relative to this script
# Adjust this path if your structure is different
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "datasets") # Points to project root/datasets

# Create datasets directory if it doesn't exist
os.makedirs(DATASETS_DIR, exist_ok=True)

# --- Pydantic Models ---
class SVDParams(BaseModel):
    resolution: str = Field("1024x576", description="Video resolution e.g., '1024x576'")
    seed: int = Field(-1, description="Random seed, -1 for random")
    motion_bucket_id: int = Field(127, description="Motion bucket ID for SVD")
    fps: int = Field(6, description="Frames per second for the generated video")
    noise_aug_strength: float = Field(0.02, description="Noise augmentation strength for SVD")
    # Add other relevant SVD parameters if needed

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
    """Load the required models when the application starts."""
    global sd_pipeline, svd_pipeline, device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        logger.warning("CUDA not available, using CPU. Generation will be very slow.")
        device = "cpu"

    logger.info(f"Using device: {device}")

    # --- Load Text-to-Image Model (Stable Diffusion v1.5) ---
    logger.info("Loading Stable Diffusion model (v1.5)...")
    try:
        from diffusers import StableDiffusionPipeline # Import here or globally
        dtype = torch.float16 if device == "cuda" else torch.float32
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            # variant="fp16" # v1.5 doesn't typically use fp16 variant folder structure like SDXL/SVD
        )
        sd_pipe.to(device)
        # Optional: Add optimizations if needed (e.g., xformers)
        # if device == "cuda":
        #     try: sd_pipe.enable_xformers_memory_efficient_attention()
        #     except ImportError: logger.warning("xformers not installed, cannot enable memory efficient attention.")
        sd_pipeline = sd_pipe
        logger.info("Stable Diffusion model loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load Stable Diffusion model: {e}")
        sd_pipeline = None

    # --- Load Image-to-Video Model (Stable Video Diffusion) ---
    logger.info("Loading Stable Video Diffusion model (img2vid-xt)...")
    try:
        dtype = torch.float16 if device == "cuda" else torch.float32
        variant = "fp16" if device == "cuda" else None

        svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=dtype,
            variant=variant
        )
        svd_pipe.to(device)
        # Optional: Enable memory optimizations
        # if device == "cuda":
        #     try: svd_pipe.enable_model_cpu_offload()
        #     except Exception as optim_error: logger.warning(f"Could not apply SVD optimizations: {optim_error}")
        svd_pipeline = svd_pipe
        logger.info("Stable Video Diffusion model loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load SVD model: {e}")
        svd_pipeline = None

    if sd_pipeline is None or svd_pipeline is None:
        logger.error("One or more models failed to load. Generation endpoint might not work.")

# --- Helper Functions ---
def get_dataset_path(dataset_name: str) -> str:
    """Constructs the path for a given dataset name."""
    safe_dataset_name = "".join(c for c in dataset_name if c.isalnum() or c in ('_', '-')).rstrip()
    if not safe_dataset_name:
        safe_dataset_name = "invalid_name"
    base_path = os.path.join(DATASETS_DIR, safe_dataset_name)
    
    # Create required subdirectories
    for subdir in ['images', 'videos', 'exports']:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
    
    return base_path

def get_media_paths(dataset_path: str, base_uuid: str) -> dict:
    """Generates paths for all media files related to a generation."""
    return {
        'initial_image': os.path.join(dataset_path, 'images', f'{base_uuid}_initial.png'),
        'video': os.path.join(dataset_path, 'videos', f'{base_uuid}.mp4'),
        'export_zip': os.path.join(dataset_path, 'exports', f'dataset_{base_uuid}.zip'),
        'export_json': os.path.join(dataset_path, 'exports', f'metadata_{base_uuid}.json')
    }

def save_initial_image(image: Image.Image, filepath: str):
    """Saves the initial generated image with metadata in EXIF."""
    try:
        # Convert to RGB if needed (some image models output RGBA)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(filepath, 'PNG')
        logger.info(f"Initial image saved to: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save initial image: {e}")
        raise

def update_metadata(dataset_path: str, metadata: dict):
    """Appends new metadata to the dataset's metadata.json file."""
    metadata_filepath = os.path.join(dataset_path, "metadata.json")
    data = []
    if os.path.exists(metadata_filepath):
        try:
            with open(metadata_filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode existing metadata file: {metadata_filepath}. Starting fresh.")
            data = [] # Reset if file is corrupt
        except Exception as e:
            logger.error(f"Error reading metadata file {metadata_filepath}: {e}")
            # Decide how to handle - maybe raise error or start fresh
            data = []

    if not isinstance(data, list):
        logger.warning(f"Metadata file {metadata_filepath} is not a list. Resetting.")
        data = [] # Ensure it's a list

    data.append(metadata)

    try:
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Metadata updated: {metadata_filepath}")
    except Exception as e:
        logger.error(f"Error writing metadata file {metadata_filepath}: {e}")
        # Handle write error (e.g., raise HTTPException)


# --- API Endpoints ---

@app.get("/api/health", tags=["Status"])
async def health_check():
    """Basic health check endpoint."""
    # TODO: Add checks for model loading status if applicable
    return {"status": "ok", "message": "API is running"}

@app.post("/api/generate", tags=["Generation"])
async def generate_video_endpoint(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Receives a prompt and parameters, generates a video using SVD (asynchronously),
    saves it to the specified dataset, and updates metadata.
    NOTE: Generation runs synchronously in this version. Consider background tasks for long generations.
    """
    global sd_pipeline, svd_pipeline, device # Use both pipelines now
    if sd_pipeline is None or svd_pipeline is None:
        logger.error("One or both pipelines (SD/SVD) are not loaded. Cannot generate video.")
        raise HTTPException(status_code=503, detail="Video generation models are not available.")

    logger.info(f"Received generation request for dataset '{request.dataset_name}': {request.prompt}")
    logger.info(f"Parameters: {request.parameters.dict()}")

    # --- Actual Generation Logic ---

    # 1. Generate Initial Image using Stable Diffusion (Text-to-Image)
    logger.info("Generating initial image using Stable Diffusion...")
    try:
        # Determine target size for SVD input (often 1024x576 or 576x1024)
        svd_width_str, svd_height_str = request.parameters.resolution.split('x')
        svd_width, svd_height = int(svd_width_str), int(svd_height_str)

        # Generate image using SD pipeline
        with torch.inference_mode():
            # Note: SD v1.5 base resolution is 512x512. Generating directly at 1024x576 might be suboptimal.
            # Consider generating at 512x512 and resizing, or using specific techniques for higher-res generation if needed.
            # For simplicity, let's try generating closer to the target aspect ratio if possible, or just 512x512.
            # Let's generate at 512x512 for now.
            sd_image_result = sd_pipeline(
                prompt=request.prompt,
                height=512, # Standard SD v1.5 height
                width=512,  # Standard SD v1.5 width
                num_inference_steps=30, # Fewer steps for faster generation, adjust as needed
                generator=torch.manual_seed(request.parameters.seed) if request.parameters.seed != -1 else None,
                # Add other SD parameters like guidance_scale if desired
            ).images[0] # Get the first generated image

        # Resize the generated image to match the SVD input resolution
        logger.info(f"Resizing initial image from {sd_image_result.width}x{sd_image_result.height} to {svd_width}x{svd_height}")
        initial_image = sd_image_result.resize((svd_width, svd_height), Image.Resampling.LANCZOS)

        logger.info("Initial image generated and resized successfully.")

    except ValueError: # Handle resolution parsing error specifically
        logger.error(f"Invalid resolution format for SVD: {request.parameters.resolution}")
        raise HTTPException(status_code=400, detail="Invalid SVD resolution format. Use 'WidthxHeight'.")
    except torch.cuda.OutOfMemoryError:
         logger.error("CUDA Out of Memory during Stable Diffusion generation.")
         raise HTTPException(status_code=500, detail="Generation failed due to insufficient GPU memory (Text-to-Image step).")
    except Exception as e:
        logger.error(f"Error during Stable Diffusion generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Initial image generation failed: {e}")


    # 2. Run SVD Pipeline (Image-to-Video) using the generated initial_image
    logger.info("Starting video generation using SVD...")
    try:
        with torch.inference_mode():
            frames = svd_pipeline( # Use svd_pipeline here
                initial_image, # Use the image generated by SD
                decode_chunk_size=8, # Adjust based on VRAM
                generator=torch.manual_seed(request.parameters.seed) if request.parameters.seed != -1 else None, # Consider if seed should be reused or different for SVD
                motion_bucket_id=request.parameters.motion_bucket_id,
                noise_aug_strength=request.parameters.noise_aug_strength,
                num_frames=25, # Default SVD frame count
                # Add other SVD parameters if needed
            ).frames[0]
        logger.info("SVD video generation finished successfully.")

    except torch.cuda.OutOfMemoryError:
         logger.error("CUDA Out of Memory during SVD generation.")
         raise HTTPException(status_code=500, detail="Generation failed due to insufficient GPU memory (Image-to-Video step).")
    except Exception as e:
        logger.error(f"Error during SVD generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Video generation failed (Image-to-Video step): {e}")

    # 3. Save Video File (No changes needed here)
    dataset_path = get_dataset_path(request.dataset_name)
    video_dir = os.path.join(dataset_path, "videos")
    os.makedirs(video_dir, exist_ok=True)

    video_filename_base = f"video_{uuid.uuid4()}"
    video_filename = f"{video_filename_base}.mp4"
    video_filepath = os.path.join(video_dir, video_filename)

    try:
        export_to_video(frames, video_filepath, fps=request.parameters.fps)
        logger.info(f"Video saved successfully to: {video_filepath}")
    except Exception as e:
        logger.error(f"Error saving video file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # 4. Prepare and Save Metadata (Update model info)
    sd_model_name = sd_pipeline.config._name_or_path if sd_pipeline and hasattr(sd_pipeline, 'config') else "Unknown SD Model"
    svd_model_name = svd_pipeline.config._name_or_path if svd_pipeline and hasattr(svd_pipeline, 'config') else "Unknown SVD Model"
    metadata = {
        "filename": video_filename,
        "prompt": request.prompt,
        "dataset_name": request.dataset_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_info": { # Store info about both models used
             "text_to_image": sd_model_name,
             "image_to_video": svd_model_name
        },
        "parameters": request.parameters.dict() # These are SVD parameters
        # TODO: Consider adding SD parameters to metadata as well if they become configurable
    }
    update_metadata(dataset_path, metadata)

    # 5. Return Response (Update message)
    video_serve_path = f"/datasets/{request.dataset_name}/videos/{video_filename}"

    return JSONResponse(content={
        "message": "Video generated successfully using Text-to-Image and Image-to-Video.",
        "video_url": video_serve_path, # URL frontend can use to fetch the video
        "metadata": metadata
    })


# --- Dataset Management Endpoints ---
@app.get("/api/datasets", tags=["Datasets"])
async def list_datasets():
    """Lists all available datasets with basic statistics."""
    try:
        datasets = []
        for item in os.listdir(DATASETS_DIR):
            dataset_path = os.path.join(DATASETS_DIR, item)
            if os.path.isdir(dataset_path):
                # Count files in each directory
                video_count = len(os.listdir(os.path.join(dataset_path, 'videos')))
                image_count = len(os.listdir(os.path.join(dataset_path, 'images')))
                
                # Get metadata file info
                metadata_path = os.path.join(dataset_path, 'metadata.json')
                metadata_exists = os.path.exists(metadata_path)
                metadata_size = os.path.getsize(metadata_path) if metadata_exists else 0
                
                datasets.append({
                    "name": item,
                    "video_count": video_count,
                    "image_count": image_count,
                    "has_metadata": metadata_exists,
                    "metadata_size": metadata_size,
                    "last_modified": datetime.fromtimestamp(os.path.getmtime(dataset_path)).isoformat()
                })
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail="Failed to list datasets")

@app.post("/api/datasets/{dataset_name}/export", tags=["Datasets"])
async def export_dataset(dataset_name: str, background_tasks: BackgroundTasks):
    """Exports a dataset as a ZIP file containing all media files and metadata."""
    import zipfile
    import shutil
    
    dataset_path = get_dataset_path(dataset_name)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Create a unique export filename
    export_id = uuid.uuid4()
    export_path = os.path.join(dataset_path, 'exports', f'dataset_{export_id}.zip')
    
    try:
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add metadata.json
            metadata_path = os.path.join(dataset_path, 'metadata.json')
            if os.path.exists(metadata_path):
                zipf.write(metadata_path, 'metadata.json')
            
            # Add all images
            for img in os.listdir(os.path.join(dataset_path, 'images')):
                img_path = os.path.join(dataset_path, 'images', img)
                zipf.write(img_path, f'images/{img}')
            
            # Add all videos
            for vid in os.listdir(os.path.join(dataset_path, 'videos')):
                vid_path = os.path.join(dataset_path, 'videos', vid)
                zipf.write(vid_path, f'videos/{vid}')
        
        export_url = f"/datasets/{dataset_name}/exports/dataset_{export_id}.zip"
        return {
            "message": "Dataset exported successfully",
            "download_url": export_url,
            "export_id": str(export_id)
        }
    except Exception as e:
        logger.error(f"Error exporting dataset: {e}")
        if os.path.exists(export_path):
            os.remove(export_path)  # Clean up failed export
        raise HTTPException(status_code=500, detail=f"Failed to export dataset: {str(e)}")

@app.delete("/api/datasets/{dataset_name}", tags=["Datasets"])
async def delete_dataset(dataset_name: str):
    """Deletes an entire dataset including all media files and metadata."""
    dataset_path = get_dataset_path(dataset_name)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        shutil.rmtree(dataset_path)
        return {"message": f"Dataset '{dataset_name}' deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

# --- Media Serving Endpoints ---
@app.get("/datasets/{dataset_name}/exports/{filename}", tags=["Data"])
async def get_export(dataset_name: str, filename: str):
    """Serves an exported dataset file."""
    dataset_path = get_dataset_path(dataset_name)
    file_path = os.path.join(dataset_path, 'exports', filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Export file not found")
    
    return FileResponse(
        file_path,
        media_type='application/zip',
        filename=filename
    )

@app.get("/datasets/{dataset_name}/images/{filename}", tags=["Data"])
async def get_image(dataset_name: str, filename: str):
    """Serves a generated initial image file."""
    dataset_path = get_dataset_path(dataset_name)
    file_path = os.path.join(dataset_path, 'images', filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        file_path,
        media_type='image/png',
        filename=filename
    )

@app.get("/datasets/{dataset_name}/videos/{video_filename}", tags=["Data"])
async def get_video(dataset_name: str, video_filename: str):
    """Serves a generated video file."""
    dataset_path = get_dataset_path(dataset_name)
    video_filepath = os.path.join(dataset_path, "videos", video_filename)

    if not os.path.exists(video_filepath):
        logger.error(f"Video file not found: {video_filepath}")
        raise HTTPException(status_code=404, detail="Video file not found")

    # Use FileResponse to stream the video
    return FileResponse(video_filepath, media_type="video/mp4", filename=video_filename)


# --- Serve Frontend ---
# Mount static files (CSS, JS) from the frontend directory
# Make sure this path is correct relative to where you run uvicorn
static_dir = os.path.join(FRONTEND_DIR, "static")
if not os.path.isdir(static_dir):
     logger.warning(f"Static directory not found at {static_dir}. Frontend static files might not load.")
     # Optionally create it: os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root():
    """Serves the main index.html file."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        logger.error(f"index.html not found at {index_path}")
        return HTMLResponse(content="<html><body><h1>Frontend not found</h1><p>Place index.html in app/frontend/</p></body></html>", status_code=404)
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error reading index.html: {e}")
        return HTMLResponse(content="<html><body><h1>Error loading frontend</h1></body></html>", status_code=500)

# --- Main Execution ---
if __name__ == "__main__":
    # Recommended way to run for development: uvicorn app.backend.main:app --reload --port 8000
    logger.info(f"Frontend directory expected at: {os.path.abspath(FRONTEND_DIR)}")
    logger.info(f"Datasets directory expected at: {os.path.abspath(DATASETS_DIR)}")
    logger.info("Starting server using uvicorn.run() (suitable for basic testing)")
    logger.info("For development with auto-reload, run: uvicorn app.backend.main:app --reload --port 8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
