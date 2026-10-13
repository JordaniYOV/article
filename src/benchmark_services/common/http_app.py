"""Authenticated private HTTP interface; models load only in job subprocesses."""

from contextlib import asynccontextmanager
import hmac
import logging
import platform
import sys
from threading import Event, Thread

from fastapi import Depends, FastAPI, Header, HTTPException

from .controller import JobController, ServiceBusy
from .locks import gpu_busy
from .protocol import BACKENDS, PROTOCOL, JobSubmission, store_identity
from .settings import Settings


def create_service(service, settings=None):
    if service not in BACKENDS:
        raise ValueError("Unknown model service")
    settings = settings or Settings()

    def authorize(authorization: str | None = Header(default=None)):
        if not settings.service_token:
            raise HTTPException(503, "Set PLANETARY_API_SERVICE_TOKEN before using model services")
        expected = "Bearer " + settings.service_token
        if not (authorization or "").isascii() or not hmac.compare_digest(authorization or "", expected):
            raise HTTPException(401, "Invalid service token")

    @asynccontextmanager
    async def lifespan(app):
        controller = JobController(service, settings)
        app.state.controller = controller
        stop = Event()
        app.state.watchdog_error = None

        def watchdog():
            while not stop.wait(min(settings.worker_poll_seconds, 1)):
                try:
                    controller.reap()
                    app.state.watchdog_error = None
                except Exception as error:
                    app.state.watchdog_error = f"{type(error).__name__}: {error}"
                    logging.getLogger(__name__).exception("Model watchdog could not reap a job")

        thread = Thread(target=watchdog, name=f"{service}-watchdog", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=2)
            controller.close()

    app = FastAPI(title=f"Planetary {service} service", lifespan=lifespan,
        dependencies=[Depends(authorize)])

    @app.get("/health")
    def health():
        controller = app.state.controller
        with controller.guard:
            controller.reap()
            return {"status": "ok", "protocol": PROTOCOL, "service": service,
                "backends": list(BACKENDS[service]), "store_id": store_identity(settings),
                "mock": settings.service_mock, "active_run_id": controller.run_id,
                "gpu_busy": gpu_busy(), "python": platform.python_version(),
                "python_executable": sys.executable, "watchdog_error": app.state.watchdog_error}

    def call(operation, *args):
        try:
            return operation(*args)
        except LookupError as error:
            raise HTTPException(404, str(error)) from error
        except ServiceBusy as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except OSError as error:
            raise HTTPException(503, f"Service could not launch the worker: {error}") from error

    @app.post("/jobs", status_code=202)
    def submit(payload: JobSubmission):
        return call(app.state.controller.submit, payload)

    @app.get("/jobs/{run_id}")
    def status(run_id: str):
        return call(app.state.controller.status, run_id)

    @app.post("/jobs/{run_id}/cancel")
    def cancel(run_id: str):
        return call(app.state.controller.cancel, run_id)

    return app
