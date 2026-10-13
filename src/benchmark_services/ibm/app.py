"""uvicorn benchmark_services.ibm.app:app --host 127.0.0.1 --port 8002"""

from ..common.http_app import create_service

app = create_service("ibm")
