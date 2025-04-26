# 🎬 GenDatasetVideo

<div align="center">

![GenDatasetVideo Logo](logogen.png)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Stable Diffusion](https://img.shields.io/badge/Stable_Diffusion-v1.5-orange.svg)](https://huggingface.co/runwayml/stable-diffusion-v1-5)
[![License](https://img.shields.io/badge/License-All_Rights_Reserved-red.svg)](LICENSE)

*Generate stunning video datasets from text descriptions using AI 🚀*

[Features](#✨-features) • [Installation](#🚀-installation) • [Usage](#📖-usage) • [Documentation](#📚-documentation)

</div>

---

## ✨ Features

<div align="center">

| 🎥 Generation | 📊 Management | 🚀 Performance |
|--------------|--------------|----------------|
| Text-to-Image with SD 1.5 | Dataset Organization | Async Processing |
| Image-to-Video with SVD | Metadata Tracking | Redis Caching |
| Custom Parameters | Export to ZIP | Celery Task Queue |
| Resolution Control | File Management | GPU Optimization |

</div>

## 🎯 System Requirements

- 🐍 Python 3.8 or higher
- 🎮 CUDA-capable GPU (recommended)
- 📦 Redis Server
- 💾 Adequate storage space

## 🚀 Installation

1️⃣ **Clone the Repository**

```bash
git clone https://github.com/yourusername/GenDatasetVideo.git
cd GenDatasetVideo
```

2️⃣ **Set Up Virtual Environment**

```bash
# Create environment
python -m venv .venv

# Activate (choose based on your OS)
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

## 📖 Usage

1️⃣ **Start Redis Server**

```bash
redis-server
```

2️⃣ **Launch Celery Worker**

```bash
celery -A app.backend.celery_app worker --loglevel=info
```

3️⃣ **Run FastAPI Server**

```bash
# Development
uvicorn app.backend.main_async:app --reload --port 8000

# Production
uvicorn app.backend.main_async:app --port 8000
```

4️⃣ **Access the Application**

- Open `http://localhost:8000` in your browser
- Start generating amazing videos! 🎉

## 🏗️ Project Structure

```
GenDatasetVideo/
├── 📁 app/
│   ├── 📁 backend/
│   │   ├── 📄 main.py           # Synchronous API
│   │   ├── 📄 main_async.py     # Async API with Celery
│   │   ├── 📄 celery_app.py     # Worker definitions
│   │   └── 📄 celeryconfig.py   # Celery settings
│   ├── 📁 frontend/
│   │   ├── 📄 index.html        # Web interface
│   │   └── 📁 static/
│   │       ├── 📄 script.js     # Frontend logic
│   │       └── 📄 style.css     # Styling
│   └── 📁 datasets/             # Generated content
```

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate video from text |
| `/api/tasks/{task_id}` | GET | Check task status |
| `/api/datasets` | GET | List all datasets |
| `/api/datasets/{name}/export` | POST | Export as ZIP |
| `/api/datasets/{name}` | DELETE | Delete dataset |
| `/datasets/{name}/videos/{file}` | GET | Get video file |
| `/datasets/{name}/images/{file}` | GET | Get image file |

## ⚙️ Configuration

| File | Purpose |
|------|---------|
| `requirements.txt` | 📦 Dependencies |
| `celeryconfig.py` | ⚡ Performance settings |
| `LICENSE` | 📜 Legal information |

## 🚀 Performance Features

- 🔄 Background task processing
- ⚡ Redis response caching
- 🎮 GPU memory optimization
- 🔒 Rate limiting (2 gen/minute)

## ⚠️ Notes

- 🎮 Requires significant GPU memory
- ⏱️ Generation may take several minutes
- 💾 Monitor storage space usage
- 🔄 Check task status regularly

## 📜 License

All Rights Reserved. See [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by JonusNattapong/zombitx64

[🔝 Back to Top](#-gendatasetvideo)

</div>
