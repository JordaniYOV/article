"""Dispatch queued runs to isolated HTTP model services; no model imports."""

from threading import Event, Thread

from sqlalchemy import update as sql_update
from sqlmodel import Session, select

from benchmark_services.common.execution import recover
from benchmark_services.common.locks import exclusive_lock, gpu_lock, gpu_busy
from benchmark_services.common.protocol import service_for_backend
from benchmark_services.common.preparation import prepare_bundle, PreparationCancelled
from benchmark_services.common.store import ModelConfig, Run, utc_now
from .service_client import ServiceClient, ServiceUnavailable, ServiceRejected
from .logging import get_logger

LOGGER = get_logger("api.dispatcher")


class Dispatcher:
    def __init__(self, engine, settings, client=None):
        self.engine, self.settings = engine, settings
        self.client = client or ServiceClient(settings)
        self.stop = Event()
        self.thread = None
        self.last_error = None

    def _mark_queued(self, run_id, **values):
        # A service may have claimed/completed a job while the HTTP response was lost.
        # Never overwrite that newer state from an old API snapshot.
        with Session(self.engine) as session:
            session.execute(sql_update(Run).where(Run.id == run_id, Run.status == "queued").values(
                **values, updated_at=utc_now()))
            session.commit()

    def tick(self):
        try:
            with exclusive_lock("dispatcher.lock"):
                return self._tick()
        except OSError as error:
            if error.errno in {11, 13} or getattr(error, "winerror", None) in {33, 36}:
                return False  # Another dispatcher owns this tick.
            raise

    def _tick(self):
        if gpu_busy():
            return False
        # Recovery is safe only under the actual GPU lock. Live job processes,
        # including orphans after a service/API restart, keep that lock held.
        try:
            with gpu_lock():
                recover(self.engine)
        except OSError:
            return False
        with Session(self.engine) as session:
            runs = session.exec(select(Run).where(Run.status == "queued").order_by(Run.created_at, Run.id)).all()
            configs = {run.id: session.get(ModelConfig, run.model_config_id) for run in runs}
        for run in runs:
            if self.stop.is_set():
                return False
            try:
                service = service_for_backend(configs[run.id].backend)
                info = self.client.health(service)
                if configs[run.id].backend not in info.get("backends", []):
                    raise ServiceUnavailable("The configured service does not support this backend")
                if info.get("gpu_busy") or info.get("active_run_id"):
                    return False
                self._mark_queued(run.id, phase="preparing", error_message=None)
                LOGGER.info("run_preparing", run_id=run.id, service=service)

                def stopped():
                    if self.stop.is_set():
                        return True
                    with Session(self.engine) as session:
                        current = session.get(Run, run.id)
                        return current.status != "queued" or current.cancel_requested

                prepare_bundle(run, stopped)
                if stopped():
                    LOGGER.info("run_dispatch_stopped", run_id=run.id, service=service)
                    return False
                result = self.client.submit(service, run)
                if result.get("status") == "queued":
                    self._mark_queued(run.id, phase="starting_service", error_message=None)
                self.last_error = None
                LOGGER.info("run_dispatched", run_id=run.id, service=service, status=result.get("status"))
                return True
            except PreparationCancelled:
                LOGGER.info("run_preparation_cancelled", run_id=run.id)
                return False
            except (ValueError, OSError, KeyError) as error:
                LOGGER.error("run_dispatch_blocked", run_id=run.id, error_type=type(error).__name__)
                self._mark_queued(run.id, status="blocked", phase="stopped", error_message=str(error))
            except ServiceRejected as error:
                if error.status == 409:  # Shared reservation lost to another submission.
                    return False
                self.last_error = str(error)
                LOGGER.warning("run_service_rejected", run_id=run.id, error_type=type(error).__name__)
                self._mark_queued(run.id, status="blocked", phase="stopped", error_message=str(error))
            except ServiceUnavailable as error:
                # Keep the job durable and allow the other model's queue to progress.
                self.last_error = str(error)
                self._mark_queued(run.id, phase="waiting_service", error_message=str(error))
        return False

    def start(self):
        if self.thread is not None:
            return

        def loop():
            while not self.stop.is_set():
                try:
                    self.tick()
                except Exception as error:
                    self.last_error = f"{type(error).__name__}: {error}"
                    LOGGER.error("dispatcher_tick_failed", error_type=type(error).__name__)
                self.stop.wait(self.settings.worker_poll_seconds)

        self.thread = Thread(target=loop, name="model-dispatcher", daemon=True)
        self.thread.start()
        LOGGER.info("dispatcher_started")

    def close(self):
        self.stop.set()
        if self.thread is not None:
            # An in-flight HTTP call is bounded. Do not dispose its DB engine early.
            self.thread.join(timeout=2 * self.settings.service_http_timeout + self.settings.worker_poll_seconds + 1)
            if self.thread.is_alive():
                raise RuntimeError("Dispatcher did not stop within its bounded shutdown window")
            LOGGER.info("dispatcher_stopped")
