#!/usr/bin/env python3
"""Verify that event dimensions match Magenta's original implementation."""

# Temporarily set up path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from performance_rnn_torch.core.sequence import EventSeq, ControlSeq

print("=" * 70)
print("Event Dimension Verification")
print("=" * 70)

feat_dims = EventSeq.feat_dims()
print("\nEvent Feature Dimensions:")
for feat_name, feat_dim in feat_dims.items():
    print(f"  {feat_name:15} = {feat_dim:3} events")

total_dim = EventSeq.dim()
print(f"\nTotal Event Dimension: {total_dim}")
print(f"Expected (Magenta):     388")
print(f"Match: {'✓ YES' if total_dim == 388 else '✗ NO'}")

print("\n" + "=" * 70)
print("Control Dimension Verification")
print("=" * 70)

ctrl_dims = ControlSeq.feat_dims()
print("\nControl Feature Dimensions:")
for feat_name, feat_dim in ctrl_dims.items():
    print(f"  {feat_name:15} = {feat_dim:3} features")

ctrl_total = ControlSeq.dim()
print(f"\nTotal Control Dimension: {ctrl_total}")

print("\n" + "=" * 70)
print("Event Ranges:")
print("=" * 70)

feat_ranges = EventSeq.feat_ranges()
for feat_name, feat_range in feat_ranges.items():
    print(f"  {feat_name:15}: [{feat_range.start:3}, {feat_range.stop:3})")

print("\n" + "=" * 70)
