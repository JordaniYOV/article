"""One subprocess per job, plus a reservation shared by both model services."""

import os
from pathlib import Path
import subprocess
import sys
from threading import RLock
import time
from uuid import uuid4

from sqlmodel import Session, SQLModel
from sqlalchemy import case, update as sql_update

from api.logging import get_logger, WorkerLogMirror

from .execution import request_cancellation
from .files import safe_path, atomic_json
from .locks import exclusive_lock, gpu_busy
from .protocol import BACKENDS, TERMINAL
from .preparation import bundle_directory
from .store import ModelConfig, Run, utc_now
from . import settings as paths


class ServiceBusy(RuntimeError):
    pass


class JobController:
    def __init__(self, service, settings):
        if service not in BACKENDS:
            raise ValueError("Unknown model service")
        self.service, self.settings = service, settings
        self.engine = settings.create_engine()
        SQLModel.metadata.create_all(self.engine)
        self.guard = RLock()
        self.process = self.run_id = self.reservation = self.log = None
        self.log_mirror = None
        self.started = 0
        self.attempt_id = None
        self.logger = get_logger("worker.controller", service=service, mock=settings.service_mock)

    def _run(self, run_id):
        with Session(self.engine) as session:
            run = session.get(Run, run_id)
            if run is None:
                raise LookupError("Run not found")
            model = session.get(ModelConfig, run.model_config_id)
            if model.backend not in BACKENDS[self.service]:
                raise ValueError("Run belongs to a different model service")
            return run

    def _stop_record(self, run_id, reason, status="interrupted"):
        with Session(self.engine) as session:
            session.execute(sql_update(Run).where(Run.id == run_id, Run.status.not_in(TERMINAL)).values(
                status=case((Run.cancel_requested.is_(True), "cancelled"), else_=status),
                phase="stopped", error_message=case((Run.cancel_requested.is_(True), None), else_=reason),
                updated_at=utc_now()))
            session.commit()

    def _release(self):
        if self.log_mirror is not None:
            # The child has exited: drain its final bytes before closing the file.
            # A broken/slow terminal must not retain the shared GPU reservation.
            self.log_mirror.close()
            self.log_mirror = None
        if self.log is not None:
            self.log.close()
        if self.reservation is not None:
            self.reservation.__exit__(None, None, None)
        self.process = self.run_id = self.reservation = self.log = None
        self.attempt_id = None

    def _was_claimed(self, run):
        import json
        marker = safe_path(run.output_dir, paths.WORKSPACE_ROOT / "outputs") / "service-attempt.json"
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
            return record.get("attempt_id") == self.attempt_id and record.get("claimed") is True
        except (OSError, ValueError):
            return False

    def reap(self):
        with self.guard:
            if self.process is None:
                return
            code = self.process.poll()
            if code is None:
                limit = self.settings.service_max_job_seconds
                if limit and time.monotonic() - self.started > limit:
                    self.terminate("Configured model job timeout exceeded")
                return
            run = self._run(self.run_id)
            self.logger.info("job_process_exited", run_id=run.id, attempt_id=self.attempt_id,
                pid=getattr(self.process, "pid", None), exit_code=code, status=run.status,
                elapsed_seconds=time.monotonic() - self.started)
            if run.status not in TERMINAL:
                # A user can resume a terminal job while its old child is exiting.
                # Preserve that new queued attempt instead of overwriting it.
                if run.status != "queued" or not self._was_claimed(run):
                    status = "blocked" if run.status == "queued" else "interrupted"
                    self._stop_record(run.id, f"Model process exited ({code}); see service-worker.log", status)
            self._release()

    def submit(self, payload):
        with self.guard:
            self.reap()
            run = self._run(payload.run_id)
            if payload.protocol_sha256 != run.protocol_sha256:
                raise ValueError("Protocol hash differs from the queued run")
            if self.run_id == run.id:
                return self.status(run.id)  # Idempotent retry after a lost HTTP response.
            if run.status != "queued" or run.cancel_requested:
                return self.status(run.id)
            if not (bundle_directory(run) / "prepared.json").is_file():
                raise ValueError("API must prepare the shared image-only bundle before submitting this job")
            if self.process is not None or gpu_busy():
                raise ServiceBusy("The shared GPU is reserved by another job")
            reservation = exclusive_lock("model-service.lock")
            try:
                reservation.__enter__()
            except OSError as error:
                raise ServiceBusy("The shared GPU is reserved by another job") from error
            log = None
            try:
                folder = safe_path(run.output_dir, paths.WORKSPACE_ROOT / "outputs")
                folder.mkdir(parents=True, exist_ok=True)
                log = (folder / "service-worker.log").open("ab", buffering=0)
                log_offset = log.tell()  # Capture before the child can append its first event.
                env = dict(os.environ)
                env["PLANETARY_WORKSPACE_ROOT"] = str(paths.WORKSPACE_ROOT)
                env["PLANETARY_API_DATABASE_PATH"] = str(self.settings.database_path)
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                attempt_id = uuid4().hex
                env["PLANETARY_SERVICE_ATTEMPT_ID"] = attempt_id
                atomic_json(folder / "service-attempt.json", {"attempt_id": attempt_id, "claimed": False})
                # Server-side, fixed executable/module; never accept commands or paths over HTTP.
                worker = Path(__file__).resolve().parents[1] / self.service / "worker.py"
                command = [sys.executable, str(worker), "--run-id", run.id]
                if self.settings.service_mock:
                    command.append("--mock")
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                try:
                    process = subprocess.Popen(command, cwd=paths.WORKSPACE_ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
                except BaseException:
                    log.close()
                    raise
                self.process, self.run_id = process, run.id
                self.reservation, self.log, self.started = reservation, log, time.monotonic()
                self.attempt_id = attempt_id
                mirror = WorkerLogMirror(folder / "service-worker.log", log_offset, sys.stdout)
                try:
                    mirror.start()
                except (RuntimeError, OSError) as error:
                    self.logger.warning("worker_console_mirror_unavailable", run_id=run.id,
                        error_type=type(error).__name__)
                else:
                    self.log_mirror = mirror
                self.logger.info("job_submitted", run_id=run.id, attempt_id=attempt_id,
                    pid=getattr(process, "pid", None), status=run.status)
            except BaseException as error:
                self.logger.error("job_submission_failed", run_id=run.id, error_type=type(error).__name__)
                if log is not None and not log.closed:
                    log.close()
                reservation.__exit__(None, None, None)
                raise
            return self.status(run.id)

    def status(self, run_id):
        with self.guard:
            self.reap()
            run = self._run(run_id)
            with Session(self.engine) as session:
                backend = session.get(ModelConfig, run.model_config_id).backend
            return {"run_id": run.id, "status": run.status, "phase": run.phase,
                "completed": run.completed, "total": run.total, "error_message": run.error_message,
                "worker_alive": self.run_id == run.id and self.process is not None,
                "mock": self.settings.service_mock or backend == "mock" or run.loading.get("mock", False) is True}

    def cancel(self, run_id):
        with self.guard:
            run = self._run(run_id)
            with Session(self.engine) as session:
                changed = request_cancellation(session, run.id)
            self.logger.info("job_cancel_requested" if changed else "job_cancel_unchanged", run_id=run.id)
            return self.status(run.id)

    def terminate(self, reason):
        with self.guard:
            if self.process is None:
                return
            self.logger.info("job_process_terminating", run_id=self.run_id, attempt_id=self.attempt_id,
                pid=getattr(self.process, "pid", None), reason="owned_process_stop")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            try:
                run = self._run(self.run_id)
                if run.status != "queued" or not self._was_claimed(run):
                    self._stop_record(self.run_id, reason)
            finally:
                self._release()

    def close(self):
        try:
            self.terminate("Model service stopped; resume from saved responses")
        finally:
            self.engine.dispose()
