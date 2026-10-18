"""Bounded authenticated HTTP to model services; never imports model adapters."""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

from planetary_vlm.network import urlopen
from benchmark_services.common.protocol import PROTOCOL, store_identity


class ServiceUnavailable(RuntimeError):
    pass


class ServiceRejected(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


class ServiceClient:
    def __init__(self, settings):
        self.settings = settings
        self.urls = {"spacellava": settings.spacellava_service_url, "ibm": settings.ibm_service_url, "heads": settings.heads_service_url}

    def request(self, service, method, path, payload=None):
        url = self.urls[service].rstrip("/")
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            raise ServiceUnavailable("Invalid configured model service URL")
        if not self.settings.service_token:
            raise ServiceUnavailable("Set PLANETARY_API_SERVICE_TOKEN")
        body = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        request = Request(url + path, data=body, method=method,
            headers={"Authorization": "Bearer " + self.settings.service_token, "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.settings.service_http_timeout) as response:
                content = response.read(128 * 1024 + 1)
                if len(content) > 128 * 1024:
                    raise ServiceUnavailable("Model service response exceeds the size limit")
                result = json.loads(content)
                if not isinstance(result, dict):
                    raise ServiceUnavailable("Invalid model service response")
                return result
        except HTTPError as error:
            with error:
                detail = error.read(4096).decode("utf-8", errors="replace")
            raise ServiceRejected(error.code, detail) from error
        except (URLError, TimeoutError, OSError, ValueError) as error:
            raise ServiceUnavailable(f"Model service unavailable: {type(error).__name__}: {error}") from error

    def health(self, service):
        info = self.request(service, "GET", "/health")
        if (info.get("protocol") != PROTOCOL or info.get("service") != service
            or info.get("store_id") != store_identity(self.settings)):
            raise ServiceUnavailable("Model service protocol/workspace/database identity mismatch")
        return info

    def submit(self, service, run):
        result = self.request(service, "POST", "/jobs",
            {"run_id": run.id, "protocol_sha256": run.protocol_sha256})
        if result.get("run_id") != run.id:
            raise ServiceUnavailable("Model service returned a different run ID")
        return result
