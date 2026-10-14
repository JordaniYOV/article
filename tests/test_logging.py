"""Operational JSON logs never include arbitrary worker/model payloads."""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from api.logging import get_logger, WorkerLogMirror
except ImportError:
    get_logger = None


@unittest.skipIf(get_logger is None, "Install requirements/api.txt")
class LoggingTests(unittest.TestCase):
    def test_json_operational_fields_timestamp_and_pid(self):
        stream = StringIO()
        with redirect_stdout(stream):
            log = get_logger("worker", run_id="selected-run", service="ibm", mock=True, pid=0)
            log.info("request_completed", request_id="request-1", completed=1, total=3,
                     elapsed_seconds=0.25, status="ok")
        event = json.loads(stream.getvalue())
        self.assertEqual(event["event"], "request_completed")
        self.assertEqual(event["pid"], os.getpid())
        self.assertEqual(event["level"], "info")
        self.assertEqual(event["run_id"], "selected-run")
        self.assertTrue(event["timestamp"].endswith("Z"))
        self.assertTrue(event["mock"])

    def test_sensitive_payloads_and_exception_messages_are_not_emitted(self):
        stream = StringIO()
        with redirect_stdout(stream):
            log = get_logger("worker", run_id="safe-run")
            try:
                raise ValueError("private-token-and-prompt")
            except ValueError:
                log.exception("request_failed", error_type="ValueError",
                    token="private-token", prompt="private-question", raw_response="private-answer",
                    targets={"answer": "YES"}, model_options={"secret": "private-secret"})
        content = stream.getvalue()
        self.assertNotIn("private", content)
        event = json.loads(content)
        self.assertEqual(event["error_type"], "ValueError")
        self.assertNotIn("exception", event)
        self.assertNotIn("exc_info", event)

    def test_debug_filtered_and_nested_operational_field_rejected(self):
        stream = StringIO()
        with redirect_stdout(stream):
            log = get_logger("worker")
            log.debug("not-emitted")
            log.info("worker_started", run_id={"token": "secret"})
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertNotIn("run_id", json.loads(lines[0]))

    def test_mirror_streams_only_new_bytes_and_drains_unicode_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.log"
            previous = b"previous attempt\n"
            path.write_bytes(previous)
            console = StringIO()
            mirror = WorkerLogMirror(path, len(previous), console)
            mirror.start()
            try:
                payload = 'worker started\nЛунный снимок\n'.encode("utf-8")
                # Include a split UTF-8 codepoint to exercise incremental decoding.
                split = len(b"worker started\n") + 1
                with path.open("ab", buffering=0) as writer:
                    writer.write(payload[:split])
                    deadline = time.monotonic() + 2
                    while "worker started" not in console.getvalue() and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertIn("worker started", console.getvalue())
                    writer.write(payload[split:])
            finally:
                self.assertTrue(mirror.close())
            self.assertEqual(console.getvalue(), payload.decode("utf-8"))
            self.assertEqual(path.read_bytes(), previous + payload)
            self.assertIsNone(mirror.error_type)

    def test_mirror_broken_console_does_not_change_file_or_hang_shutdown(self):
        class BrokenConsole:
            def write(self, text): raise BrokenPipeError("fixture console closed")
            def flush(self): pass

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.log"
            path.write_bytes(b"kept in file\n")
            mirror = WorkerLogMirror(path, 0, BrokenConsole())
            mirror.start()
            self.assertTrue(mirror.close())
            self.assertEqual(mirror.error_type, "BrokenPipeError")
            with path.open("ab") as writer:
                writer.write(b"worker keeps logging\n")
            self.assertEqual(path.read_bytes(), b"kept in file\nworker keeps logging\n")


if __name__ == "__main__":
    unittest.main()
