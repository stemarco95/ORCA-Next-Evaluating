from datetime import datetime
import csv
import os
import time
import threading
import queue


class AuditLogger:
    enabled = True
    debug = True

    _initialized = False
    _queue = queue.Queue()
    _thread = None
    _stop_event = threading.Event()

    # -------------------------------------------------
    # Initialization
    # -------------------------------------------------

    @classmethod
    def _init(cls):
        if cls._initialized:
            return

        os.makedirs("logs", exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cls.log_file = f"logs/audit_log_{timestamp_str}.csv"

        with open(cls.log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event_type", "details"])

        cls._thread = threading.Thread(
            target=cls._logger_loop,
            daemon=True
        )
        cls._thread.start()

        cls._initialized = True

    # -------------------------------------------------
    # Logger thread
    # -------------------------------------------------

    @classmethod
    def _logger_loop(cls):
        while not cls._stop_event.is_set() or not cls._queue.empty():
            try:
                timestamp, event_type, detail = cls._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with open(cls.log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"{timestamp:.3f}", event_type, detail])

            if cls.debug:
                print(f"[{event_type.upper()}] {detail}")

            cls._queue.task_done()

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    @classmethod
    def log_event(cls, event_type: str, **kwargs):
        if not cls.enabled:
            return

        cls._init()

        timestamp = time.time()
        detail = ", ".join(f"{k}={v}" for k, v in kwargs.items())

        cls._queue.put((timestamp, event_type, detail))

    @classmethod
    def log_message_sent(cls, topic: str, sender: str):
        cls.log_event("message_sent", topic=topic, sender=sender)

    @classmethod
    def log_intervention(cls, by: str, reason: str, **kwargs):
        cls.log_event("intervention", by=by, reason=reason, **kwargs)

    @classmethod
    def log_module_execution(cls, module_id: str):
        cls.log_event("module_execution", module=module_id)

    @classmethod
    def log_message(cls, message: str):
        cls.log_event("info", msg=message)

    # -------------------------------------------------
    # Shutdown (optional, but clean)
    # -------------------------------------------------

    @classmethod
    def shutdown(cls):
        cls._stop_event.set()
        if cls._thread:
            cls._thread.join(timeout=1)