"""uvicorn benchmark_services.heads.app:app --host 127.0.0.1 --port 8003"""
from ..common.http_app import create_service

app = create_service("heads")
