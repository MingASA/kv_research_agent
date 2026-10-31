"""Synthetic CPU fixtures exercise evidence rejection, never benchmark results."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_native_results import AuditError, _metric_output, read_records, telemetry_output, verify_allocation, verify_backend


def _record(number: int, phase: str = "profiling") -> dict:
    metrics = {name: {"value": value} for name, value in {
        "output_token_count": 3, "output_sequence_length": 3, "usage_completion_tokens": 3,
        "input_sequence_length": 8, "usage_prompt_tokens": 8, "usage_total_tokens": 11,
        "request_latency": 20, "time_to_first_token": 10, "inter_token_latency": 5,
        "decode_duration": 10,
    }.items()}
    return {"metadata": {"x_request_id": f"request-{number}", "conversation_id": f"session-{number}",
                         "turn_index": 0, "benchmark_phase": phase, "was_cancelled": False,
                         "context_overflow_skip": False, "request_start_ns": number * 100 + 1,
                         "request_end_ns": number * 100 + 10}, "metrics": metrics}


class NativeEvidenceTests(unittest.TestCase):
    def _read(self, rows: list[dict]) -> dict:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "records.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            return read_records(path, 2, 1)

    def test_phase_counts_and_server_token_consistency(self) -> None:
        phases = self._read([_record(0, "warmup"), _record(1), _record(2)])
        self.assertEqual(len(phases["profiling"]), 2)

    def test_rejects_missing_request_id_even_when_counts_match(self) -> None:
        rows = [_record(0, "warmup"), _record(1), _record(2)]
        del rows[2]["metadata"]["x_request_id"]
        with self.assertRaisesRegex(AuditError, "request ID"):
            self._read(rows)

    def test_rejects_duplicate_request_id_even_when_counts_match(self) -> None:
        rows = [_record(0, "warmup"), _record(1), _record(2)]
        rows[2]["metadata"]["x_request_id"] = rows[1]["metadata"]["x_request_id"]
        with self.assertRaisesRegex(AuditError, "request ID"):
            self._read(rows)

    def test_rejects_server_client_token_disagreement(self) -> None:
        rows = [_record(0, "warmup"), _record(1), _record(2)]
        rows[2]["metrics"]["usage_completion_tokens"]["value"] = 4
        with self.assertRaisesRegex(AuditError, "completion counts"):
            self._read(rows)

    def test_rejects_duplicate_source_session(self) -> None:
        rows = [_record(0, "warmup"), _record(1), _record(2)]
        rows[2]["metadata"]["conversation_id"] = rows[1]["metadata"]["conversation_id"]
        with self.assertRaisesRegex(AuditError, "Duplicate conversation"):
            self._read(rows)

    def test_rejects_phase_count_mismatch(self) -> None:
        with self.assertRaisesRegex(AuditError, "record count"):
            self._read([_record(0, "warmup"), _record(1)])

    def test_rejects_reused_warmup_source(self) -> None:
        rows = [_record(0, "warmup"), _record(1), _record(2)]
        rows[1]["metadata"]["conversation_id"] = rows[0]["metadata"]["conversation_id"]
        with self.assertRaisesRegex(AuditError, "overlap"):
            self._read(rows)

    def test_capture_without_replay_is_not_accepted(self) -> None:
        backend = {"class": "FlashInferBackend", "attention_impl": "flashinfer", "planner_backend": "fa2",
                   "paged_prefill": True, "quality_transform": False, "decode_tensor_cores": True,
                   "graph_enabled": True, "graph_runner": "DecodeCudaGraphRunner", "kv_cache_dtype": "auto",
                   "flashinfer_version": "0.6.18.post1", "graph_captured_batches": [1, 2, 4]}
        text = "NATIVE_KV_BACKEND " + json.dumps(backend)
        with self.assertRaisesRegex(AuditError, "replay evidence"):
            verify_backend(text, backend, "bf16")
        replay = {"class": "FlashInferBackend", "batch_size": 2, "captured_batch_size": 2}
        accepted = verify_backend(text + "\nNATIVE_KV_GRAPH_REPLAY " + json.dumps(replay), backend, "bf16")
        self.assertEqual(accepted["replay_markers"], [replay])
        replay["captured_batch_size"] = 8
        with self.assertRaisesRegex(AuditError, "captured native bucket"):
            verify_backend(text + "\nNATIVE_KV_GRAPH_REPLAY " + json.dumps(replay), backend, "bf16")

    def test_allocation_is_computed_from_tensor_shapes_and_dtypes(self) -> None:
        tensor = {"dtype": "torch.bfloat16", "shape": [1, 128, 8, 128], "bytes": 262144}
        layer = {"key_format": "bf16", "value_format": "bf16", "resident_bytes": 524288,
                 "bytes_per_token": 4096, "tensors": {"key": tensor, "value": copy.deepcopy(tensor)}}
        allocation = {"layers": [copy.deepcopy(layer) for _ in range(48)],
                      "resident_bytes": 48 * 524288, "token_capacity": 128}
        self.assertEqual(verify_allocation(allocation, "bf16")["bytes_per_token_per_layer"], 4096)
        allocation["layers"][3]["tensors"]["key"]["bytes"] -= 1
        with self.assertRaisesRegex(AuditError, "tensor byte"):
            verify_allocation(allocation, "bf16")

    def test_throughput_uses_observation_interval_not_benchmark_duration_metric(self) -> None:
        rows = [_record(1), _record(2)]
        rows[0]["metadata"]["request_start_ns"] = 100_000_000
        rows[0]["metadata"]["request_end_ns"] = 200_000_000
        rows[1]["metadata"]["request_start_ns"] = 300_000_000
        rows[1]["metadata"]["request_end_ns"] = 12_100_000_000
        summary = {"start_time": "2026-10-25T00:00:00", "end_time": "2026-10-25T00:00:12",
                   "benchmark_duration": {"avg": 9}, "output_token_throughput": {"avg": 0.5},
                   "request_throughput": {"avg": 2 / 12}}
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp)
            (case / "artifacts").mkdir()
            (case / "artifacts/profile_export.jsonl").write_text("\n".join(map(json.dumps, rows)))
            result = _metric_output(case, rows, summary, 4)
        self.assertEqual(result["output_tps"], 0.5)
        self.assertEqual(result["profile_observation_duration_s"], 12)
        self.assertEqual(result["benchmark_duration_metric_s"], 9)

    def test_gpu_profile_peak_excludes_warmup_samples(self) -> None:
        row = _record(1)
        row["metadata"]["request_start_ns"] = 1_000_000_000
        row["metadata"]["request_end_ns"] = 4_000_000_000
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "telemetry.csv"
            path.write_text("utc,memory_used_mib,compute_util_pct,memory_util_pct\n"
                            "1970-01-01T00:00:00+00:00,9999,100,100\n"
                            "1970-01-01T00:00:02+00:00,500,80,20\n"
                            "1970-01-01T00:00:03+00:00,600,90,30\n"
                            "1970-01-01T00:00:05+00:00,8888,100,100\n")
            result = telemetry_output(path, [row])
        self.assertEqual(result["gpu_memory_used_mib_max"], 600)
        self.assertEqual(result["gpu_profile_samples"], 2)
        self.assertEqual(result["gpu_telemetry_max_gap_s"], 1)


if __name__ == "__main__":
    unittest.main()
