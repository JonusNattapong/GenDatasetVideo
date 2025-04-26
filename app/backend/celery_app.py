from celery import Celery
import os
import json
import logging
from datetime import datetime
import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline, StableDiffusionPipeline
from diffusers.utils import export_to_video

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery('video_generation_tasks')
celery_app.config_from_object('app.backend.celeryconfig')

# Optional: Configure Celery logging
celery_app.conf.worker_log_format = "[%(asctime)s: %(levelname)s/%(processName)s] %(message)s"
celery_app.conf.worker_task_log_format = (
    "[%(asctime)s: %(levelname)s/%(processName)s] "
    "[%(task_name)s(%(task_id)s)] %(message)s"
)

# Global variables for ML models
sd_pipeline = None
svd_pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    """Load ML models if not already loaded."""
    global sd_pipeline, svd_pipeline, device
    
    try:
        if sd_pipeline is None:
            logger.info("Loading Stable Diffusion model...")
            dtype = torch.float16 if device == "cuda" else torch.float32
            sd_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype
            ).to(device)
            
        if svd_pipeline is None:
            logger.info("Loading Stable Video Diffusion model...")
            dtype = torch.float16 if device == "cuda" else torch.float32
            variant = "fp16" if device == "cuda" else None
            svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=dtype,
                variant=variant
            ).to(device)
            
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

@celery_app.task(bind=True)
def generate_video_task(self, request_data):
    """Celery task for generating video from text prompt."""
    try:
        # Ensure models are loaded
        if not load_models():
            raise Exception("Failed to load ML models")

        # Extract parameters
        prompt = request_data['prompt']
        dataset_name = request_data['dataset_name']
        parameters = request_data['parameters']
        
        # Update task state
        self.update_state(state='PROCESSING', meta={'status': 'Generating initial image...'})
        
        # 1. Generate initial image with Stable Diffusion
        with torch.inference_mode():
            sd_image = sd_pipeline(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=30,
                generator=torch.manual_seed(parameters['seed']) if parameters['seed'] != -1 else None,
            ).images[0]
        
        # Resize for SVD
        width, height = map(int, parameters['resolution'].split('x'))
        initial_image = sd_image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Update task state
        self.update_state(state='PROCESSING', meta={'status': 'Generating video...'})
        
        # 2. Generate video with SVD
        with torch.inference_mode():
            frames = svd_pipeline(
                initial_image,
                decode_chunk_size=8,
                generator=torch.manual_seed(parameters['seed']) if parameters['seed'] != -1 else None,
                motion_bucket_id=parameters['motion_bucket_id'],
                noise_aug_strength=parameters['noise_aug_strength'],
                num_frames=25
            ).frames[0]
        
        # Update task state
        self.update_state(state='SAVING', meta={'status': 'Saving files...'})
        
        # 3. Save files and metadata
        from pathlib import Path
        base_path = Path(f"datasets/{dataset_name}")
        base_uuid = f"gen_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(parameters['seed'])}"
        
        # Save initial image
        images_dir = base_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_path = images_dir / f"{base_uuid}_initial.png"
        initial_image.save(image_path)
        
        # Save video
        videos_dir = base_path / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        video_path = videos_dir / f"{base_uuid}.mp4"
        export_to_video(frames, str(video_path), fps=parameters['fps'])
        
        # Prepare metadata
        metadata = {
            "id": base_uuid,
            "prompt": prompt,
            "dataset_name": dataset_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "parameters": parameters,
            "files": {
                "initial_image": str(image_path.relative_to(base_path)),
                "video": str(video_path.relative_to(base_path))
            }
        }
        
        # Save metadata
        metadata_path = base_path / "metadata.json"
        try:
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
            else:
                existing_metadata = []
        except json.JSONDecodeError:
            existing_metadata = []
        
        existing_metadata.append(metadata)
        with open(metadata_path, 'w') as f:
            json.dump(existing_metadata, f, indent=2)
        
        return {
            "status": "success",
            "metadata": metadata,
            "paths": {
                "image": str(image_path),
                "video": str(video_path)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in generate_video_task: {e}")
        raise
