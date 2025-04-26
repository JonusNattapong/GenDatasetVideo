"""
Celery configuration file.
Run worker with: celery -A app.backend.celery_app worker --loglevel=info
"""

# --- Broker Settings ---
broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

# --- Task Settings ---
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'UTC'
enable_utc = True

# --- Task Execution Settings ---
task_track_started = True
task_time_limit = 3600  # 1 hour max runtime
task_soft_time_limit = 3000  # 50 minutes soft limit
worker_max_tasks_per_child = 10  # Restart worker after 10 tasks
worker_prefetch_multiplier = 1  # Don't prefetch tasks (one at a time due to GPU memory)

# --- Memory Management ---
worker_max_memory_per_child = 4000000  # 4GB memory limit per worker
task_annotations = {
    'app.backend.celery_app.generate_video_task': {
        'rate_limit': '2/m'  # Max 2 generations per minute
    }
}

# --- Redis Cache Settings ---
cache_backend = 'redis'
cache_backend_options = {
    'host': 'localhost',
    'port': 6379,
    'db': 1,  # Use different DB than broker
    'socket_timeout': 3,
}

# --- Task Queues ---
task_default_queue = 'default'
task_queues = {
    'default': {
        'exchange': 'default',
        'routing_key': 'default',
    },
    'generate': {  # Special queue for generation tasks
        'exchange': 'generate',
        'routing_key': 'generate',
    }
}

# --- Routing ---
task_routes = {
    'app.backend.celery_app.generate_video_task': {'queue': 'generate'},
}

# --- Error Handling ---
task_eager_propagates = True  # Propagate errors in testing/debug
task_store_errors_even_if_ignored = True
