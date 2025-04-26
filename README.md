# GenDatasetVideo

A web application for generating video datasets using text prompts, powered by Stable Diffusion and Stable Video Diffusion.

## Features

- **Text-to-Video Generation:** Generate videos from text descriptions using:
  - Text-to-Image (Stable Diffusion v1.5)
  - Image-to-Video (Stable Video Diffusion img2vid-xt)
- **Dataset Management:** Organize generated content into datasets
- **Asynchronous Processing:** Background task processing for long-running generations
- **Performance Optimizations:** Redis caching and Celery task queue
- **Export Capabilities:** Export datasets as ZIP files with metadata

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Redis Server

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GenDatasetVideo.git
cd GenDatasetVideo
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start Redis Server:
```bash
redis-server
```

2. Start Celery Worker:
```bash
celery -A app.backend.celery_app worker --loglevel=info
```

3. Start the FastAPI Server:
```bash
# Development with auto-reload
uvicorn app.backend.main_async:app --reload --port 8000
# Production
uvicorn app.backend.main_async:app --port 8000
```

4. Open your browser and navigate to `http://localhost:8000`

## Project Structure

```
app/
├── backend/
│   ├── main.py              # Synchronous API implementation
│   ├── main_async.py        # Asynchronous API with Celery
│   ├── celery_app.py        # Celery worker and task definitions
│   └── celeryconfig.py      # Celery configuration
├── frontend/
│   ├── index.html          # Web interface
│   └── static/
│       ├── script.js       # Frontend JavaScript
│       └── style.css       # CSS styles
└── datasets/               # Generated content
    └── <dataset_name>/
        ├── images/         # Generated images
        ├── videos/         # Generated videos
        ├── exports/        # Dataset exports
        └── metadata.json   # Dataset metadata
```

## API Endpoints

- `POST /api/generate` - Generate video from text prompt
- `GET /api/tasks/{task_id}` - Check task status
- `GET /api/datasets` - List all datasets
- `POST /api/datasets/{name}/export` - Export dataset as ZIP
- `DELETE /api/datasets/{name}` - Delete dataset
- `GET /datasets/{name}/videos/{file}` - Get video file
- `GET /datasets/{name}/images/{file}` - Get image file

## Configuration

Key configuration files:
- `requirements.txt` - Python dependencies
- `app/backend/celeryconfig.py` - Celery & Redis settings
- `LICENSE` - License information

## Performance Features

- Background task processing with Celery
- Redis caching for improved response times
- Resource management and rate limiting
- Memory optimization for GPU usage

## License

All Rights Reserved. See LICENSE file for details.

## Notes

- The application requires significant GPU memory for video generation
- Default rate limit: 2 generations per minute
- Video generation may take several minutes depending on hardware
- Ensure adequate disk space for dataset storage
