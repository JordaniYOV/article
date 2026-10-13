"""uvicorn benchmark_services.spacellava.app:app --host 127.0.0.1 --port 8001"""

from ..common.http_app import create_service

app = create_service("spacellava")
