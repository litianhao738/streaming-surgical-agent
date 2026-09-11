"""One synchronous queue writer for budget snapshots; no filesystem/API retries."""
from concurrent.futures import Future
from copy import deepcopy
from queue import Queue
from threading import Thread

from surgical_agent.artifacts.manifest import atomic_write_json


class SerializedLedgerMixin:
    """Take snapshots under the transport RLock and serialize writes on one thread."""

    def persist(self):
        with self.lock:
            if getattr(self, "_ledger_closed", False):
                raise RuntimeError("ledger writer already closed")
            if not hasattr(self, "_ledger_queue"):
                self._ledger_queue = Queue()
                self._ledger_thread = Thread(target=self._write_ledgers, daemon=True,
                                             name="budget-single-writer")
                self._ledger_thread.start()
            snapshot = deepcopy({
                "limits": {k: str(v) for k, v in self.limits.items()},
                "carried_occupied": self.carried,
                "occupied": {k: str(v) for k, v in self.occupied.items()},
                "calls": self.rows, "stopped": self.stopped,
            })
            completed = Future()
            self._ledger_queue.put((snapshot, completed))
            # Writer never acquires self.lock. Dispatch waits for durable reservation.
            completed.result()

    def _write_ledgers(self):
        while True:
            item = self._ledger_queue.get()
            if item is None:
                return
            snapshot, completed = item
            try:
                atomic_write_json(self.output / "budget.json", snapshot)
            except Exception as exc:  # noqa: BLE001 -- deliver writer failures to caller
                completed.set_exception(exc)
            else:
                completed.set_result(None)

    def close_ledger(self):
        with self.lock:
            if getattr(self, "_ledger_closed", False):
                return
            if hasattr(self, "_ledger_queue"):
                self._ledger_queue.put(None)
                self._ledger_thread.join()
            self._ledger_closed = True
