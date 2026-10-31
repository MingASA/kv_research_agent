#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Check FP8 representation algebra on CPU; no model calibration or GPU claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def analyze() -> dict:
    values = torch.arange(0x7F80, dtype=torch.int32, device="cpu").to(torch.int16).view(torch.bfloat16).float()
    rows = []
    for label, scale in (("key", .125), ("value", .0625)):
        for alternative in (scale / 2, scale * 2):
            current = (values / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
            other = (values / alternative).clamp(-448, 448).to(torch.float8_e4m3fn).float() * alternative
            common = ((values >= max(scale, alternative) * 2**-6)
                      & (values <= min(scale, alternative) * 448))
            count = int(common.sum())
            differences = int((current[common] != other[common]).sum())
            if count == 0 or differences:
                raise ValueError("Unexpected power-of-two normal-domain representation change")
            rows.append({"cache": label, "current_scalar": scale,
                         "alternative_power_of_two_scalar": alternative,
                         "positive_BF16_values_in_common_normal_unsaturated_domain": count,
                         "different_dequantized_values_in_that_domain": differences,
                         "current_max_finite": 448 * scale,
                         "current_smallest_nonzero_subnormal": 2**-9 * scale,
                         "current_smallest_normal": 2**-6 * scale})
    return {"classification": "confirmed_CPU_FP8_representation_algebra_not_model_calibration",
            "scope": "All positive finite BF16 bit patterns; FP8 E4M3 RNE QDQ, no model activations or GPU timing",
            "torch_version": torch.__version__, "device": "cpu", "rows": rows,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inference": "Power-of-two scalar changes preserve dequantized values in the shared normal "
                         "unsaturated domain; investigate observed clipping/underflow before assuming calibration "
                         "will improve model quality.",
            "limitations": ["Datatype values are not production activation frequencies",
                            "This does not locate observed quality losses or certify a scale",
                            "Actual GPU writer/kernel rounding and activation statistics remain separate evidence"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = analyze()
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"classification": result["classification"], "checks": len(result["rows"]), "device": "cpu"}))


if __name__ == "__main__":
    main()
