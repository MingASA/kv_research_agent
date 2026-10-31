# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Bounded observations of the already admitted worker's private xLLM server.

Call only inside the owned network-none serving container after its lease,
identity and server readiness checks. This is no host or GPU profiler.
"""

from __future__ import annotations

import datetime
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
from typing import Callable


METRICS = {
    "num_pending_requests": "request_gauge",
    "num_running_requests": "request_gauge",
    "num_waiting_requests": "request_gauge",
    "num_preempted_requests": "iteration_gauge_not_cumulative",
    "num_running_sequences": "sequence_gauge",
    "num_free_blocks": "block_gauge",
    "num_used_blocks": "effective_used_block_gauge",
    "num_blocks_in_prefix_cache": "prefix_block_gauge_may_be_reclaimable",
    "kv_cache_utilization_perc": "fraction_used_over_pages_minus_one_includes_padding_in_used",
    "prefix_cache_prompt_tokens_total": "cumulative_tokens",
    "prefix_cache_hit_tokens_total": "cumulative_tokens",
    "prefix_cache_token_hit_rate_perc": "percent_0_to_100",
}


class _Value(HTMLParser):
    def __init__(self, name: str) -> None:
        super().__init__(convert_charrefs=True)
        self._target = "value-" + name
        self._inside = False
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "span" and dict(attrs).get("id") == self._target:
            self._inside = True
            self.values.append("")

    def handle_data(self, data: str) -> None:
        if self._inside:
            self.values[-1] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "span":
            self._inside = False


def parse_metric(name: str, body: str) -> float:
    if name not in METRICS:
        raise ValueError("Metric outside the explicit application observation list")
    parsed = _Value(name)
    parsed.feed(body)
    if len(parsed.values) != 1:
        raise ValueError("Missing or ambiguous actual application value")
    value = float(parsed.values[0].strip())
    if not math.isfinite(value) or value < 0:
        raise ValueError("Application value is negative or nonfinite")
    # BlockManager reserves page0 via allocate(): used includes that page but
    # its denominator is physical_pages-1. Thus a full cache can exceed1.
    # For at least2 physical pages the mathematical upper bound is2.
    maximum = 2 if name == "kv_cache_utilization_perc" else 100 if name == "prefix_cache_token_hit_rate_perc" else None
    if maximum is not None and value > maximum:
        raise ValueError("Application metric unit/range differs from its source contract")
    if "gauge" in METRICS[name] or METRICS[name] == "cumulative_tokens":
        if not value.is_integer():
            raise ValueError("Count metric is not integral")
    return value


class ServiceMetricCollector:
    """Sample explicit bvars; failures remain visible in the immutable JSONL."""

    def __init__(self, output: Path, run_id: str, request_sha256: str, physical_pages: int,
                 on_sample: Callable[[dict], None] | None = None) -> None:
        if type(physical_pages) is not int or physical_pages < 2:
            raise ValueError("Observed allocator physical page capacity is required")
        self._output, self._run_id, self._request_sha = output, run_id, request_sha256
        self._physical_pages = physical_pages
        self._on_sample = on_sample
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None

    def _run(self) -> None:
        try:
            with self._output.open("x", buffering=1) as stream:
                stream.write(json.dumps({"schema": "kv-owned-service-metrics-v1", "run_id": self._run_id,
                    "request_sha256": self._request_sha, "worker_pid": os.getpid(), "units": METRICS,
                    "scope": "private_owned_container_loopback_only", "sample_interval_seconds": 1,
                    "physical_pages": self._physical_pages, "reserved_padding_pages": 1,
                    "limitation": "Sequential application observations, not atomic snapshots or HBM hardware counters"}) + "\n")
                while not self._stop.is_set():
                    start = time.monotonic()
                    row = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "monotonic_ns": time.monotonic_ns(), "values": {}, "errors": {}}
                    for name in METRICS:
                        if self._stop.is_set():
                            break
                        try:
                            with urllib.request.urlopen("http://127.0.0.1:19001/vars/" + name, timeout=.2) as response:
                                body = response.read(128 * 1024 + 1)
                            if len(body) > 128 * 1024:
                                raise ValueError("Application metric exceeded bounded response size")
                            row["values"][name] = parse_metric(name, body.decode())
                        except (OSError, ValueError, urllib.error.URLError) as error:
                            row["errors"][name] = type(error).__name__ + ": " + str(error)[:160]
                    row["sample_elapsed_seconds"] = time.monotonic() - start
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    if self._on_sample is not None:
                        self._on_sample(row)
                    self._stop.wait(max(0, 1 - row["sample_elapsed_seconds"]))
                stream.flush()
                os.fsync(stream.fileno())
        except (OSError, ValueError) as error:
            self._failure = error

    def start(self) -> None:
        if self._thread is not None or self._output.exists():
            raise ValueError("Application metrics require a new single writer/output")
        self._thread = threading.Thread(target=self._run, name="owned-service-metrics", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Owned application observation did not stop in its bounded deadline")
        if self._failure is not None:
            raise RuntimeError("Application observation failed; original output retained") from self._failure
