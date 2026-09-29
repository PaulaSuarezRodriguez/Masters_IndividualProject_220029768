"""
fedma_resnet.py
===============

Federated Matched Averaging (FedMA) for ResNet-style CNNs, aggregation only.

    Wang, Yurochkin, Sun, Papailiopoulos, Khazaeni.
    "Federated Learning with Matched Averaging", ICLR 2020.

READ THIS FIRST - what FedMA is, and what is left out here
-----------------------------------------------------------
FedMA has two halves:

  (1) Matched averaging.  Solve eq. 2/3 with BBP-MAP to find permutations,
      then average matched channels:  W_n = (1/J) * sum_j W_{j,n} Pi_j^T.
  (2) Layer-wise training.  Match layer 1 only, broadcast it to the clients,
      have them RETRAIN every later layer with layer 1 frozen, then match
      layer 2, retrain again, and so on (Algorithm 1).

You asked for aggregation only, so this module implements (1) and omits (2).
That omission is not cosmetic, and the FedMA paper is blunt about it: the
authors show (Section 2.3, Figure 1) that matched averaging WITHOUT the
retraining loop "fails on deep architectures", and that applying BBP-MAP
straight through a deep CNN "only gives very minor improvements over weight
averaging".  The layer-wise retraining IS the contribution.

So in aggregation-only form, FedMA and PFNM differ only in four concrete
details, all implemented below:

  a. Atom definition.  FedMA matches on INCOMING weights: for CNN layer n the
     atoms are the rows of Pi_{j,n-1}^T W_{j,n} flattened over C_in x w x h
     (eq. 5).  PFNM's top-down scheme uses OUTGOING weights instead.
  b. Ordering.  FedMA is recursive front-to-back: layer n-1's permutation is
     fixed before layer n is matched (eq. 4/5, recursion base Pi_{j,0} = I,
     the identity on the input image).  PFNM sweeps top-down.
  c. Aggregation.  FedMA takes the plain 1/J mean of matched channels
     (Algorithm 1).  PFNM takes the Gaussian posterior mean, which shrinks
     toward the prior.
  d. Last layer.  FedMA does not match it; it averages weighted by each
     client's class proportions p_jk (Algorithm 1, else branch).

Everything here is forward-pass free: no gradients, no training.

What FedMA does not cover, and this module has to invent
--------------------------------------------------------
The paper ignores BatchNorm outright ("we ignore all batch normalization
layers in the VGG architecture and leave it for future work") and lists
residual connections as future work in the conclusion.  A ResNet-20 is built
from both.  The handling of residual streams and BN here is inherited from
pfnm_resnet.py and is my construction, not the paper's.  Every permutation is
checked to be function preserving before it is used.

Requires pfnm_resnet.py beside it.  That is deliberate: FedMA's own solver is
BBP-MAP, which the FedMA paper takes unchanged from PFNM, so the solver, the
ResNet permutation-group analysis and the BN recalibration are shared rather
than duplicated.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from pfnm_resnet import (  # shared machinery
    build_resnet_perm_spec,
    permute_state_dict,
    verify_permutation_invariance,
    recalibrate_batchnorm,
    match_group,          # BBP-MAP solver (FedMA eq. 2/3 = PFNM Prop. 2/3)
    _atoms_for_group,
    _derive_coupled,
    _inv,
    ResNetPermSpec,
)

__all__ = [
    "fedma_group_order",
    "fedma_merge_state_dicts",
    "concatenate_classifiers",
    "recalibrate_batchnorm",
]


# =============================================================================
# FedMA's recursive layer ordering
# =============================================================================

def fedma_group_order(spec: ResNetPermSpec) -> List[str]:
    """Order the permutation groups front-to-back, as FedMA's recursion requires.

    FedMA matches layer n only after layer n-1 is fixed, so that the atoms of
    layer n are Pi_{j,n-1}^T W_{j,n} - i.e. already aligned on their input side
    (eq. 4/5).  The recursion base is the input image, whose ordering is fixed.

    A ResNet is not a chain, so a strict layer order does not exist: a residual
    stream is written to by the stem AND by the last conv of every block in its
    stage, at several different depths.  The order used here is the closest
    topological approximation:

        stream(stage 1)                      <- written by the stem, input = image
        block-internal groups of stage 1
        block-internal groups of stage 2, stream(stage 2)
        block-internal groups of stage 3, stream(stage 3)
        ...

    The one place this is imperfect is the first stream: it is also written by
    the conv2 of every stage-1 block, whose input groups are matched *after* it.
    Those contributions are therefore aligned by the identity on the first pass.
    Running a second pass (passes=2) fixes it, at the cost of no longer being a
    single front-to-back recursion.
    """
    order: List[str] = []
    streams = [g for g in spec.stream_groups if g not in spec.coupled_streams]
    all_streams = list(spec.stream_groups)

    def stage_of(group: str) -> str:
        if group.startswith("stream_"):
            return group[len("stream_"):]
        return group.split(".")[0]

    stages = []
    for g in all_streams:
        s = stage_of(g)
        if s not in stages:
            stages.append(s)

    first_stage = stages[0] if stages else None

    # the stream fed directly by the stem goes first
    for g in all_streams:
        if stage_of(g) == first_stage and g in streams:
            order.append(g)

    for si, stage in enumerate(stages):
        for g in spec.hidden_groups:
            if stage_of(g) == stage and g not in order:
                order.append(g)
        if si > 0:
            for g in all_streams:
                if stage_of(g) == stage and g in streams and g not in order:
                    order.append(g)

    # anything the walk missed (unusual topologies)
    for g in spec.group_size:
        if g not in order and g not in spec.coupled_streams:
            order.append(g)
    return order


# =============================================================================
# FedMA merge
# =============================================================================

def fedma_merge_state_dicts(
    state_dicts: List["OrderedDict[str, torch.Tensor]"],
    model_factory=None,
    sigma: float = 1.0,
    sigma0: float = 1.0,
    gamma: float = 7.0,
    it: int = 5,
    passes: int = 1,
    allow_growth: bool = False,
    last_layer_keys: Sequence[str] = ("linear.weight", "linear.bias",
                                      "fc.weight", "fc.bias"),
    class_proportions: Optional[Sequence[Sequence[float]]] = None,
    exclude_keys: Sequence[str] = (),
    buffer_plain_mean: bool = True,
    permute_streams: bool = True,
    verify: bool = True,
    verbose: bool = True,
    seed: int = 0,
) -> Tuple["OrderedDict[str, torch.Tensor]", Dict]:
    """Aggregate J ResNet state_dicts with FedMA's matched averaging.

    Parameters
    ----------
    sigma, sigma0, gamma
        BBP-MAP hyper-parameters.  FedMA's Appendix D.2 uses sigma=1, sigma0=1
        and gamma0=7 for VGG-9 on CIFAR-10 (PFNM used gamma0=1).  NOTE: gamma
        only controls how eagerly new global channels are created, so it has no
        effect at all unless allow_growth=True.
    passes
        1 = FedMA's single front-to-back recursion (the faithful setting).
        >1 repeats the walk, which helps the residual streams but is no longer
        a pure recursion.
    allow_growth
        FedMA lets the global model grow, max_j L_j <= L <= sum_j L_j; the
        paper reports 1.11x growth for VGG-9.  A fixed ResNet-20 cannot hold a
        wider model, so this defaults to False and the function raises a clear
        error if a group would grow.
    exclude_keys
        Parameters EXCLUDED from aggregation entirely.  They are still permuted
        (a classifier must follow the reordering of the feature channels it
        reads, or it ends up wired to the wrong features), but they do not
        appear in the returned merged state_dict.  Each model's permuted copy
        comes back in info['permuted_excluded'].  Use this to keep the
        classifier out of the merge and assemble it yourself.
    class_proportions
        (J, K) array; entry [j][k] is the fraction of the data points with
        label k that live on client j (FedMA Algorithm 1, last layer).  Leave
        as None for non-overlapping specialists: each client then owns 100% of
        its own classes and 0% of everyone else's, so the rule reduces to
        summing the zero-padded classifiers, which is what None does.
        Ignored for any key also listed in exclude_keys.
    exclude_keys
        Parameters that are NOT aggregated at all and are simply absent from
        the returned state_dict.  Pass the classifier keys here to aggregate
        only the feature extractor and build the classifier yourself with
        concatenate_classifiers().  The excluded parameters are still used for
        the permutation-invariance self-check, which is worth keeping: it
        verifies that the classifier's input channels stay aligned.
    """
    rng = np.random.RandomState(seed)
    J = len(state_dicts)
    if J < 2:
        raise ValueError("Need at least two models to merge.")

    ref = state_dicts[0]
    for j, sd in enumerate(state_dicts[1:], 1):
        for k, v in ref.items():
            if k not in sd:
                raise ValueError(f"Model {j} is missing parameter '{k}'.")
            if sd[k].shape != v.shape:
                raise ValueError(
                    f"Shape mismatch for '{k}': model 0 has {tuple(v.shape)}, "
                    f"model {j} has {tuple(sd[k].shape)}.")

    spec = build_resnet_perm_spec(ref)
    if verbose:
        print("  FedMA permutation groups discovered from the state_dict:")
        print(spec.describe())

    order = fedma_group_order(spec)
    if not permute_streams:
        order = [g for g in order if g not in spec.stream_groups]
    if verbose:
        print(f"  FedMA matching order (front-to-back): {' -> '.join(order)}")

    assignments: List[Dict[str, List[int]]] = [
        {g: list(range(n)) for g, n in spec.group_size.items()} for _ in range(J)
    ]

    info = {"groups": OrderedDict(spec.group_size),
            "shortcut_style": spec.shortcut_style,
            "matched_groups": list(order), "order": list(order),
            "passes": passes, "sigma": sigma, "sigma0": sigma0, "gamma": gamma,
            "allow_growth": allow_growth, "skipped_groups": [],
            "global_sizes": {}, "aggregation": "fedma_matched_averaging"}

    # ---- FedMA's recursion: match a group only once its inputs are fixed ----
    for p in range(passes):
        n_changed = 0
        for g in order:
            atoms, ok = [], True
            for j in range(J):
                # eq. 5: atoms are the rows of Pi_{j,n-1}^T W_{j,n}, flattened
                # over C_in x w x h  -> incoming weights, input side already
                # permuted by the groups matched earlier in the walk.
                a = _atoms_for_group(state_dicts[j], spec, g, assignments[j],
                                     mode="incoming", exclude_keys=exclude_keys)
                if a is None or a.shape[1] == 0:
                    ok = False
                    break
                atoms.append(a)
            if not ok:
                if p == 0:
                    warnings.warn(f"Group '{g}' has no incoming-weight atoms; left unpermuted.")
                    info["skipped_groups"].append(g)
                continue

            assign, _ = match_group(atoms, sigma=sigma, sigma0=sigma0, gamma=gamma,
                                    it=it, allow_growth=allow_growth, rng=rng)

            gsize = max(max(a) + 1 for a in assign)
            info["global_sizes"][g] = gsize
            if gsize != spec.group_size[g]:
                raise RuntimeError(
                    f"FedMA grew group '{g}' from {spec.group_size[g]} to {gsize} "
                    f"channels. FedMA allows this (the paper reports 1.11x growth "
                    f"for VGG-9) but a fixed ResNet-20 cannot represent it. "
                    f"Use allow_growth=False (default), or lower gamma.")

            for j in range(J):
                if assign[j] != assignments[j][g]:
                    n_changed += 1
                assignments[j][g] = list(assign[j])

        for j in range(J):
            _derive_coupled(assignments[j], spec)

        if verbose:
            print(f"    pass {p + 1}/{passes}: {n_changed} group/model assignments updated")
        if n_changed == 0:
            break

    # ---- self-check: a permutation must not change what a model computes ----
    if verify and model_factory is not None:
        all_ok = True
        for j in range(J):
            ok, err = verify_permutation_invariance(model_factory, state_dicts[j],
                                                    assignments[j], spec)
            if not ok:
                all_ok = False
                if verbose:
                    print(f"    [!] permutation invariance FAILED for model {j} "
                          f"(max |dy| = {err:.3e})")
        info["permutation_invariance_ok"] = all_ok
        if all_ok and verbose:
            print("    permutation invariance verified for all models (functions unchanged)")
        if not all_ok:
            if permute_streams:
                warnings.warn("Residual-stream permutations are not function preserving "
                              "here; falling back to block-internal channels only.")
                return fedma_merge_state_dicts(
                    state_dicts, model_factory=model_factory, sigma=sigma,
                    sigma0=sigma0, gamma=gamma, it=it, passes=passes,
                    allow_growth=allow_growth, last_layer_keys=last_layer_keys,
                    class_proportions=class_proportions, exclude_keys=exclude_keys,
                    buffer_plain_mean=buffer_plain_mean, permute_streams=False,
                    verify=verify, verbose=verbose, seed=seed)
            raise RuntimeError("FedMA produced a non-function-preserving permutation.")

    # ---- matched averaging:  W_n = (1/J) sum_j W_{j,n} Pi_j^T --------------
    permuted = [permute_state_dict(state_dicts[j], assignments[j], spec) for j in range(J)]

    merged = OrderedDict()
    for k, v in ref.items():
        if k in exclude_keys:
            continue                         # not aggregated; caller supplies it
        stack = torch.stack([permuted[j][k].float() for j in range(J)])

        if k in last_layer_keys:
            # Algorithm 1, else branch: W_N = sum_k sum_j p_jk W_{jl,N}
            if class_proportions is None:
                out = stack.sum(dim=0)          # disjoint classes -> p_jk in {0,1}
            else:
                p = torch.as_tensor(class_proportions, dtype=torch.float32)
                if p.shape[0] != J or p.shape[1] != stack.shape[1]:
                    raise ValueError(
                        f"class_proportions must be ({J}, {stack.shape[1]}), got {tuple(p.shape)}")
                w = p.reshape(J, p.shape[1], *([1] * (stack.dim() - 2)))
                out = (stack * w).sum(dim=0)
        elif k.endswith("num_batches_tracked"):
            out = stack.max(dim=0).values
        else:
            out = stack.mean(dim=0)             # plain 1/J mean
        merged[k] = out.to(v.dtype)

    info["assignments"] = assignments
    info["spec"] = spec
    info["excluded_keys"] = list(exclude_keys)
    info["permuted_excluded"] = [
        OrderedDict((k, permuted[j][k].clone()) for k in exclude_keys if k in permuted[j])
        for j in range(J)
    ]
    return merged, info


# =============================================================================
# Building the global classifier by concatenation (no aggregation)
# =============================================================================

def concatenate_classifiers(
    weights: Sequence[torch.Tensor],
    biases: Sequence[torch.Tensor],
    assignments: Sequence[Dict[str, List[int]]],
    spec: ResNetPermSpec,
    classifier_key: str = "linear.weight",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack the specialists' classifiers into one, WITHOUT aggregating them.

    The rows are simply concatenated in the order given, so specialist 0 owns
    output indices 0..n_0-1, specialist 1 owns n_0..n_0+n_1-1, and so on.

    The columns are NOT simply concatenated.  A classifier reads the channels
    of the last residual stream, and matching re-indexes those channels, so
    each specialist's classifier must first be permuted by its OWN assignment:

        W_j_aligned = W_j[:, inv(Pi_j)]

    Skipping this would leave every specialist's classifier reading features
    that have moved, which is silently wrong - the shapes still line up.

    Parameters
    ----------
    weights, biases : the specialists' original linear.weight / linear.bias,
        with their own row counts (e.g. 6, 9, 3, 25).
    assignments : the per-model assignment dicts returned in info["assignments"].
    spec : the ResNetPermSpec returned in info["spec"].
    """
    g = spec.axis1.get(classifier_key)
    if g is None and spec.stream_groups:
        g = spec.stream_groups[-1]

    out_w, out_b = [], []
    for j, (w, b) in enumerate(zip(weights, biases)):
        w = w.detach().float()
        b = b.detach().float()
        if g is not None and g in assignments[j]:
            inv = _inv(assignments[j][g])
            if w.shape[1] == len(inv):
                w = w.index_select(1, inv)
            else:
                raise ValueError(
                    f"Classifier {j} has {w.shape[1]} input features but group "
                    f"'{g}' has {len(inv)} channels.")
        out_w.append(w)
        out_b.append(b)

    return torch.cat(out_w, dim=0), torch.cat(out_b, dim=0)
