"""Crash-released cross-process locks for a single-host/shared-volume GPU."""

from contextlib import contextmanager
import os

from . import settings as paths


@contextmanager
def exclusive_lock(name):
    if name not in {"api-worker.lock", "model-service.lock", "dispatcher.lock"}:
        raise ValueError("Unknown coordination lock")
    lock_path = paths.WORKSPACE_ROOT / ".state" / name
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    locked = False
    try:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def gpu_lock(settings=None):
    return exclusive_lock("api-worker.lock")


def gpu_busy():
    for name in ("model-service.lock", "api-worker.lock"):
        try:
            with exclusive_lock(name):
                pass
        except BlockingIOError:
            return True
        except OSError as error:
            if error.errno in {11, 13} or getattr(error, "winerror", None) in {33, 36}:
                return True
            raise
    return False
