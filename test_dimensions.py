#!/usr/bin/env python3
"""Quick test to verify event dimensions match Magenta."""

print("Testing Event Dimensions...")
print("=" * 70)

try:
    from performance_rnn_torch.core.sequence import EventSeq, ControlSeq

    # Get dimensions
    feat_dims = EventSeq.feat_dims()
    total_dim = EventSeq.dim()

    print("\nEvent Feature Dimensions:")
    for feat_name, feat_dim in feat_dims.items():
        print(f"  {feat_name:15} = {feat_dim:3} events")

    print(f"\nTotal Event Dimension: {total_dim}")
    print(f"Expected (Magenta):     388")

    if total_dim == 388:
        print("\n✓ SUCCESS: Dimensions match Magenta's original implementation!")
    else:
        print(f"\n✗ FAILED: Expected 388 events, got {total_dim}")
        print("\nBreakdown:")
        print("  Expected: note_on(88) + note_off(88) + velocity(32) + time_shift(100) + set_tempo(80) = 388")

    # Also check control dimensions
    ctrl_dims = ControlSeq.feat_dims()
    ctrl_total = ControlSeq.dim()
    print(f"\nControl Dimension: {ctrl_total}")
    print(f"  (pitch_histogram: 12 + note_density: {ctrl_dims['note_density']})")

    print("\n" + "=" * 70)
    print("Test completed successfully!")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
