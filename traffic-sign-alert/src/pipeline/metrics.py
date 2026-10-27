"""
metrics.py

Captures exactly the three numbers your Evaluation Metrics slide names:
  - % of traffic signs accurately recognized
  - time taken for recognition of a traffic sign  (recognition latency)
  - time taken to determine action + output modality  (decision latency)

Kept separate from run_pipeline.py so you can unit test the accounting
logic without needing a real model loaded.
"""
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class RunMetrics:
    total_signs_seen: int = 0
    correctly_recognized: int = 0
    recognition_latencies_ms: list = field(default_factory=list)
    decision_latencies_ms: list = field(default_factory=list)

    def record_recognition(self, correct: bool, latency_ms: float):
        self.total_signs_seen += 1
        if correct:
            self.correctly_recognized += 1
        self.recognition_latencies_ms.append(latency_ms)

    def record_decision(self, latency_ms: float):
        self.decision_latencies_ms.append(latency_ms)

    @property
    def accuracy(self) -> float:
        if self.total_signs_seen == 0:
            return 0.0
        return self.correctly_recognized / self.total_signs_seen

    def summary(self) -> dict:
        def _avg(values):
            return round(statistics.mean(values), 2) if values else None

        return {
            "signs_seen": self.total_signs_seen,
            "accuracy_pct": round(self.accuracy * 100, 2),
            "avg_recognition_latency_ms": _avg(self.recognition_latencies_ms),
            "avg_decision_latency_ms": _avg(self.decision_latencies_ms),
        }


@contextmanager
def timer_ms():
    """Usage:
        with timer_ms() as t:
            ... do work ...
        elapsed = t()   # milliseconds, valid only after the `with` block exits
    """
    start = time.perf_counter()
    result = {}

    def _elapsed():
        return result["ms"]

    yield _elapsed
    result["ms"] = (time.perf_counter() - start) * 1000
