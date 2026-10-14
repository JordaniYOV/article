"""Image/answer-free JSON logging shared by API and isolated workers.

This module imports neither the HTTP application nor any model runtime. Only
operational, allowlisted scalar fields are emitted. Exception messages, local
variables, prompts, targets, responses, credentials and arbitrary payloads must
stay outside these logs. Workers' stdout is saved by their service controller.
"""

import logging
import os
import codecs
from pathlib import Path
from threading import Event, Thread

import structlog


FIELDS = frozenset({
    "event", "level", "timestamp", "component", "service", "run_id", "request_id",
    "backend", "mock", "attempt_id", "pid", "phase", "status", "completed", "total",
    "pending", "elapsed_seconds", "exit_code", "error_type", "reason", "claimed",
})


def operational_fields(logger, method, event):
    """Fail closed for payloads, exception details and unexpected log fields."""
    return {key: value for key, value in event.items()
            if key in FIELDS and (value is None or isinstance(value, (str, int, float, bool)))}


def get_logger(component, **context):
    """Create an INFO JSON logger without changing uvicorn/root logging handlers."""
    return structlog.wrap_logger(
        # Emit each JSON record + newline with one write so the mirrored stream
        # cannot be inserted between a controller record and its newline.
        structlog.WriteLogger(),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            operational_fields,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    ).bind(**{**context, "component": component, "pid": os.getpid()})


class WorkerLogMirror:
    """Tail only this attempt's bytes to the service console, never pipe the child.

    The worker writes directly to its durable file even if the console is broken
    or slow. Reads close their file handle before console I/O, so shutdown never
    leaves a run file locked by a blocked terminal. External output is mirrored
    verbatim (with UTF-8 replacement for invalid bytes), not passed as log fields.
    """

    def __init__(self, path, offset, console):
        self.path, self.offset, self.console = Path(path), offset, console
        self.stop = Event()
        self.error_type = None
        self.thread = Thread(target=self._follow, name="worker-log-console", daemon=True)

    def start(self):
        self.thread.start()

    def close(self, timeout=2):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    def _write(self, text):
        if text:
            self.console.write(text)
            self.console.flush()

    def _follow(self):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                with self.path.open("rb") as source:
                    source.seek(self.offset)
                    chunk = source.read(64 * 1024)
                if chunk:
                    self.offset += len(chunk)
                    self._write(decoder.decode(chunk))
                elif self.stop.is_set():
                    self._write(decoder.decode(b"", final=True))
                    return
                else:
                    self.stop.wait(.05)
        except Exception as error:
            # Mirroring must not crash the model job or disturb its original log.
            self.error_type = type(error).__name__
