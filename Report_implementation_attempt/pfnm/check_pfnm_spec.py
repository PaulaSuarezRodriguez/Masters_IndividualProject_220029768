"""
check_pfnm_spec.py
==================

Run this ONCE, before trusting the merged model.

It answers the only question that can silently ruin a PFNM merge on a ResNet:
"did the engine correctly work out which channels are welded together by the
residual connections in *your* ResNet20_DynamicModel?"

The test: take one of your specialists, apply a random valid permutation to
it, and check that the permuted network computes exactly the same function.
Permuting hidden units is supposed to be a pure relabelling, so if the maximum
output difference is not ~1e-7, the permutation groups are wrong for your
architecture and the merge would be garbage.

    python check_pfnm_spec.py
"""

import numpy as np
import torch

from Report_implementation_attempt.ResNet20_DynamicModel_vReport import ResNet20_DynamicModel
from pfnm_resnet import (build_resnet_perm_spec, verify_permutation_invariance,
                         _derive_coupled)

MODEL_PATH = "../final_specialised_models/final_outdoors_model.pt"


def random_assignment(spec, seed=0):
    rng = np.random.RandomState(seed)
    a = {}
    for g, n in spec.group_size.items():
        a[g] = list(range(n)) if g in spec.coupled_streams else list(rng.permutation(n))
    _derive_coupled(a, spec)
    return a


def main():
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    n_classes = state_dict["linear.weight"].shape[0]
    print(f"Loaded {MODEL_PATH}  (classifier size {n_classes})\n")

    spec = build_resnet_perm_spec(state_dict)
    print("Permutation groups discovered:")
    print(spec.describe())
    print()

    # every parameter should be accounted for
    unassigned = [k for k in state_dict
                  if spec.axis0.get(k) is None and spec.axis1.get(k) is None
                  and not k.endswith("num_batches_tracked")
                  and k not in ("linear.bias", "fc.bias")]
    if unassigned:
        print("[!] parameters not attached to any permutation group:")
        for k in unassigned:
            print(f"      {k}  {tuple(state_dict[k].shape)}")
        print("    (fine if they are genuinely unpermutable, suspicious otherwise)\n")

    all_ok = True
    for seed in range(5):
        a = random_assignment(spec, seed=seed)
        ok, err = verify_permutation_invariance(
            lambda: ResNet20_DynamicModel(num_classes=n_classes),
            state_dict, a, spec)
        print(f"  seed {seed}: {'PASS' if ok else 'FAIL'}   max|dy| = {err:.3e}")
        all_ok &= ok

    print()
    if all_ok:
        print("=" * 70)
        print("PASS - the permutation groups are correct for your architecture.")
        print("You can trust the PFNM merge.")
        print("=" * 70)
    else:
        print("=" * 70)
        print("FAIL - permuting your model changes what it computes, so the")
        print("residual-stream grouping is wrong for your architecture.")
        print()
        print("Most likely cause: your shortcut is neither a 1x1 conv nor the")
        print("standard zero-pad shortcut. Fix options, in order of preference:")
        print("  1. Tell me what your shortcut looks like and I'll fix the spec.")
        print("  2. As a safe fallback, set permute_streams=False in the call to")
        print("     pfnm_merge_state_dicts(). Only block-internal channels get")
        print("     matched, which is weaker but always correct. (The engine")
        print("     already falls back to this automatically when its own")
        print("     verification fails.)")
        print("=" * 70)


if __name__ == "__main__":
    main()
