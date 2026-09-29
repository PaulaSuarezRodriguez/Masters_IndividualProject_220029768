"""
pfnm_resnet.py
==============

Probabilistic Federated Neural Matching (PFNM) for ResNet-style CNNs.

    Yurochkin, Agarwal, Ghosh, Greenewald, Hoang, Khazaeni.
    "Bayesian Nonparametric Federated Learning of Neural Networks", ICML 2019.

The cost function (Prop. 2 / Prop. 3), the Hungarian assignment, the
Beta-Bernoulli / IBP popularity term and the Gaussian posterior mean
(Prop. 1 / Prop. 4) are taken directly from the authors' reference
implementation (``pfnm.py``, IBM/probabilistic-federated-neural-matching).

What had to be added, because the reference code only handles MLPs
--------------------------------------------------------------------
1. Convolutional layers.  A filter is the analogue of a hidden neuron: it is
   permutation invariant in exactly the same way, so it becomes a PFNM "atom".
2. BatchNorm.  BN parameters are per-channel, so they are *carried along* by
   the permutation of the channel group that produces them - they are never
   matched on their own.
3. Residual connections.  This is the part the paper explicitly lists as
   future work.  In a ResNet, ``out = block(x) + shortcut(x)``, so the channel
   ordering of a block's output is welded to the ordering of the shortcut.
   Every tensor that writes into the same residual stream must therefore share
   one single permutation.  This module discovers those groups from the
   state_dict and enforces the constraint.
4. Coordinate descent over groups.  The paper's top-down sweep assumes a
   feed-forward chain.  A ResNet is a DAG, so groups are matched by repeated
   sweeps (which is also what ``match_layer``'s ``it`` loop does internally).

Everything here is forward-pass free: no gradients, no training.

Author's-code equivalence note
------------------------------
With ``allow_growth=False`` (the default) the nonparametric branch of the cost
matrix is switched off, i.e. the gamma -> 0 limit in which the global model
keeps the local width.  That is required to stay inside a fixed ResNet-20
architecture.  Set ``allow_growth=True`` only if your architecture can be
instantiated at the inferred (larger) widths.
"""

from __future__ import annotations

import re
import warnings
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

__all__ = [
    "ResNetPermSpec",
    "build_resnet_perm_spec",
    "pfnm_merge_state_dicts",
    "permute_state_dict",
    "verify_permutation_invariance",
]


# =============================================================================
# PART 1.  PFNM core - adapted from the authors' pfnm.py
# =============================================================================

def row_param_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j):
    """Verbatim from pfnm.py."""
    match_norms = ((weights_j_l + global_weights) ** 2 / (sigma_inv_j + global_sigmas)).sum(axis=1) - (
        global_weights ** 2 / global_sigmas).sum(axis=1)
    return match_norms


def compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j,
                 prior_mean_norm, prior_inv_sigma, popularity_counts, gamma, J,
                 allow_growth=True, max_global=700):
    """pfnm.py's compute_cost, with the nonparametric branch made optional.

    allow_growth=False  ->  max_added = 0, i.e. no new global neurons may be
    created, so the global model keeps the local layer width.
    """
    Lj = weights_j.shape[0]
    counts = np.minimum(np.array(popularity_counts, dtype=np.float64), 10)
    param_cost = np.array(
        [row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j) for l in range(Lj)]
    )
    # IBP popularity term  2*log(m / (J - m))  (the factor 2 is folded in later
    # by the symmetry of the expression in the paper; pfnm.py uses log()).
    denom = np.maximum(J - counts, 1e-12)
    param_cost = param_cost + np.log(counts / denom)

    L = global_weights.shape[0]
    if allow_growth:
        max_added = min(Lj, max(max_global - L, 1))
    else:
        max_added = 0

    if max_added <= 0:
        return param_cost

    nonparam_cost = np.outer(
        (((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1)
         - (prior_mean_norm ** 2 / prior_inv_sigma).sum()),
        np.ones(max_added),
    )
    cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    nonparam_cost = nonparam_cost - cost_pois
    nonparam_cost = nonparam_cost + 2 * np.log(gamma / J)

    return np.hstack((param_cost, nonparam_cost))


def matching_upd_j(weights_j, global_weights, sigma_inv_j, global_sigmas,
                   prior_mean_norm, prior_inv_sigma, popularity_counts, gamma, J,
                   allow_growth=True, max_global=700):
    """pfnm.py's matching_upd_j, with allow_growth threaded through."""
    L = global_weights.shape[0]

    full_cost = compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j,
                             prior_mean_norm, prior_inv_sigma, popularity_counts,
                             gamma, J, allow_growth=allow_growth, max_global=max_global)

    row_ind, col_ind = linear_sum_assignment(-full_cost)

    assignment_j = []
    new_L = L

    for l, i in zip(row_ind, col_ind):
        if i < L:
            popularity_counts[i] += 1
            assignment_j.append(i)
            global_weights[i] += weights_j[l]
            global_sigmas[i] += sigma_inv_j
        else:  # new neuron
            popularity_counts += [1]
            assignment_j.append(new_L)
            new_L += 1
            global_weights = np.vstack((global_weights, prior_mean_norm + weights_j[l]))
            global_sigmas = np.vstack((global_sigmas, prior_inv_sigma + sigma_inv_j))

    return global_weights, global_sigmas, popularity_counts, assignment_j


def match_group(atoms: List[np.ndarray], sigma: float, sigma0: float, gamma: float,
                it: int = 5, allow_growth: bool = False, max_global: int = 700,
                rng: Optional[np.random.RandomState] = None):
    """PFNM matching for one permutation group.

    This is pfnm.py's ``match_layer`` with the layer bookkeeping stripped out
    (that part is ResNet-specific and lives in Part 2 below).

    Parameters
    ----------
    atoms : list of (L_j, D) arrays, one per model.
    Returns (assignment, global_atoms) where assignment[j][l] = global index.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    J = len(atoms)
    D = atoms[0].shape[1]

    sigma_inv_layer = [np.ones(D) / sigma for _ in range(J)]
    sigma_inv_prior = np.ones(D) / sigma0
    mean_prior = np.zeros(D)

    group_order = sorted(range(J), key=lambda x: -atoms[x].shape[0])

    batch_weights_norm = [w * s for w, s in zip(atoms, sigma_inv_layer)]
    prior_mean_norm = mean_prior * sigma_inv_prior

    global_weights = prior_mean_norm + batch_weights_norm[group_order[0]]
    global_sigmas = np.outer(np.ones(global_weights.shape[0]),
                             sigma_inv_prior + sigma_inv_layer[group_order[0]])

    popularity_counts = [1] * global_weights.shape[0]
    assignment = [[] for _ in range(J)]
    assignment[group_order[0]] = list(range(global_weights.shape[0]))

    # ---- initialise ----
    for j in group_order[1:]:
        global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(
            batch_weights_norm[j], global_weights, sigma_inv_layer[j], global_sigmas,
            prior_mean_norm, sigma_inv_prior, popularity_counts, gamma, J,
            allow_growth=allow_growth, max_global=max_global)
        assignment[j] = assignment_j

    # ---- iterate (leave-one-out group descent) ----
    for _ in range(it):
        random_order = rng.permutation(J)
        for j in random_order:
            to_delete = []
            Lj = len(assignment[j])
            for l, i in sorted(zip(range(Lj), assignment[j]), key=lambda x: -x[1]):
                popularity_counts[i] -= 1
                if popularity_counts[i] == 0:
                    del popularity_counts[i]
                    to_delete.append(i)
                    for j_clean in range(J):
                        for idx, l_ind in enumerate(assignment[j_clean]):
                            if i < l_ind and j_clean != j:
                                assignment[j_clean][idx] -= 1
                else:
                    global_weights[i] = global_weights[i] - batch_weights_norm[j][l]
                    global_sigmas[i] -= sigma_inv_layer[j]

            global_weights = np.delete(global_weights, to_delete, axis=0)
            global_sigmas = np.delete(global_sigmas, to_delete, axis=0)

            global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(
                batch_weights_norm[j], global_weights, sigma_inv_layer[j], global_sigmas,
                prior_mean_norm, sigma_inv_prior, popularity_counts, gamma, J,
                allow_growth=allow_growth, max_global=max_global)
            assignment[j] = assignment_j

    global_atoms = global_weights / global_sigmas
    return assignment, global_atoms


# =============================================================================
# PART 2.  Mapping a ResNet state_dict onto PFNM permutation groups
# =============================================================================

_LAYER_RE = re.compile(r"^(layer\d+)\.(\d+)\.")
_CONV_RE = re.compile(r"^(layer\d+)\.(\d+)\.conv(\d+)\.weight$")
_BN_RE = re.compile(r"^(layer\d+)\.(\d+)\.bn(\d+)\.(weight|bias|running_mean|running_var|num_batches_tracked)$")
_SC_RE = re.compile(r"^(layer\d+)\.(\d+)\.(shortcut|downsample)\.(\d+)\.(weight|bias|running_mean|running_var|num_batches_tracked)$")

_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


class ResNetPermSpec:
    """Which permutation group indexes which axis of which parameter."""

    def __init__(self):
        self.group_size: "OrderedDict[str, int]" = OrderedDict()
        self.axis0: Dict[str, Optional[str]] = {}
        self.axis1: Dict[str, Optional[str]] = {}
        self.stream_groups: List[str] = []
        self.hidden_groups: List[str] = []
        # coupled_streams[g] = (source_group, offset): g's permutation is *derived*
        # from source_group because the stage transition uses a parameter-free
        # zero-padding shortcut rather than a 1x1 conv.
        self.coupled_streams: "OrderedDict[str, Tuple[str, int]]" = OrderedDict()
        self.shortcut_style: str = "unknown"

    # -- convenience lookups -------------------------------------------------
    def params_produced_by(self, g: str) -> List[str]:
        return [k for k, v in self.axis0.items() if v == g]

    def params_consumed_by(self, g: str) -> List[str]:
        return [k for k, v in self.axis1.items() if v == g]

    @property
    def free_groups(self) -> List[str]:
        """Groups whose permutation is solved for directly by PFNM."""
        return [g for g in self.group_size
                if g not in self.coupled_streams]

    def describe(self) -> str:
        lines = [f"  shortcut style : {self.shortcut_style}",
                 f"  groups         : {len(self.group_size)} "
                 f"({len(self.stream_groups)} residual streams, {len(self.hidden_groups)} block-internal)"]
        for g, n in self.group_size.items():
            tag = ""
            if g in self.coupled_streams:
                src, off = self.coupled_streams[g]
                tag = f"   <- derived from {src} (offset {off})"
            lines.append(f"      {g:<28} size={n:<4}{tag}")
        return "\n".join(lines)


def build_resnet_perm_spec(state_dict: "OrderedDict[str, torch.Tensor]",
                           classifier_keys: Sequence[str] = ("linear", "fc"),
                           stem_conv: str = "conv1.weight",
                           stem_bn: str = "bn1") -> ResNetPermSpec:
    """Derive the PFNM permutation groups of a ResNet from its state_dict.

    Residual-stream groups
        Every tensor that writes into the same residual stream (the stem, each
        block's last conv/BN, and the projection shortcut) shares one group,
        because ``out = block(x) + shortcut(x)`` only stays valid if all of
        them are permuted identically.

    Block-internal groups
        The channels between a block's first and last conv are private to that
        block and get their own independent group.
    """
    spec = ResNetPermSpec()
    keys = list(state_dict.keys())

    # ---- discover stages and blocks ------------------------------------
    stages: "OrderedDict[str, List[int]]" = OrderedDict()
    for k in keys:
        m = _LAYER_RE.match(k)
        if m:
            stage, blk = m.group(1), int(m.group(2))
            stages.setdefault(stage, [])
            if blk not in stages[stage]:
                stages[stage].append(blk)
    if not stages:
        raise ValueError("No 'layerX.Y.' parameters found - is this a ResNet state_dict?")
    stage_names = sorted(stages.keys(), key=lambda s: int(s.replace("layer", "")))
    for s in stage_names:
        stages[s] = sorted(stages[s])

    # ---- residual stream groups ----------------------------------------
    # stream size = out-channels of the last conv of the stage's first block
    stream_of_stage: Dict[str, str] = {}
    for si, stage in enumerate(stage_names):
        blk = stages[stage][0]
        conv_ids = sorted(int(_CONV_RE.match(k).group(3)) for k in keys
                          if _CONV_RE.match(k) and _CONV_RE.match(k).group(1) == stage
                          and int(_CONV_RE.match(k).group(2)) == blk)
        last_conv = f"{stage}.{blk}.conv{conv_ids[-1]}.weight"
        g = f"stream_{stage}"
        spec.group_size[g] = state_dict[last_conv].shape[0]
        spec.stream_groups.append(g)
        stream_of_stage[stage] = g

    # ---- stem ------------------------------------------------------------
    first_stream = stream_of_stage[stage_names[0]]
    if stem_conv in state_dict:
        spec.axis0[stem_conv] = first_stream
        spec.axis1[stem_conv] = None  # RGB input: never permuted
    for k in keys:
        if k.startswith(stem_bn + "."):
            spec.axis0[k] = first_stream
            spec.axis1[k] = None

    # ---- blocks ----------------------------------------------------------
    for si, stage in enumerate(stage_names):
        out_stream = stream_of_stage[stage]
        prev_stream = stream_of_stage[stage_names[si - 1]] if si > 0 else first_stream

        for bi, blk in enumerate(stages[stage]):
            prefix = f"{stage}.{blk}"
            in_stream = prev_stream if bi == 0 else out_stream

            conv_ids = sorted(int(_CONV_RE.match(k).group(3)) for k in keys
                              if _CONV_RE.match(k) and _CONV_RE.match(k).group(1) == stage
                              and int(_CONV_RE.match(k).group(2)) == blk)
            n_conv = len(conv_ids)

            prev_g = in_stream
            for ci, cid in enumerate(conv_ids):
                ckey = f"{prefix}.conv{cid}.weight"
                is_last = (ci == n_conv - 1)
                if is_last:
                    g = out_stream
                else:
                    g = f"{prefix}.h{cid}"
                    spec.group_size[g] = state_dict[ckey].shape[0]
                    spec.hidden_groups.append(g)

                spec.axis0[ckey] = g
                spec.axis1[ckey] = prev_g

                for k in keys:
                    m = _BN_RE.match(k)
                    if m and m.group(1) == stage and int(m.group(2)) == blk and int(m.group(3)) == cid:
                        spec.axis0[k] = g
                        spec.axis1[k] = None
                prev_g = g

            # ---- shortcut ------------------------------------------------
            sc_keys = [k for k in keys if _SC_RE.match(k)
                       and _SC_RE.match(k).group(1) == stage
                       and int(_SC_RE.match(k).group(2)) == blk]
            if sc_keys:
                spec.shortcut_style = "conv"
                for k in sc_keys:
                    t = state_dict[k]
                    spec.axis0[k] = out_stream
                    spec.axis1[k] = in_stream if t.dim() >= 2 else None
            elif bi == 0 and si > 0:
                # No shortcut parameters at a stage transition -> parameter-free
                # pad/identity shortcut (akamaster "option A").
                if spec.shortcut_style == "unknown":
                    spec.shortcut_style = "pad"
                n_out, n_in = spec.group_size[out_stream], spec.group_size[prev_stream]
                offset = (n_out - n_in) // 2
                spec.coupled_streams[out_stream] = (prev_stream, offset)

    if spec.shortcut_style == "unknown":
        spec.shortcut_style = "identity"

    # ---- classifier -------------------------------------------------------
    last_stream = stream_of_stage[stage_names[-1]]
    for base in classifier_keys:
        wk, bk = f"{base}.weight", f"{base}.bias"
        if wk in state_dict:
            spec.axis0[wk] = None          # output classes: handled by class remapping
            spec.axis1[wk] = last_stream
        if bk in state_dict:
            spec.axis0[bk] = None
            spec.axis1[bk] = None

    # anything not registered is left untouched
    for k in keys:
        spec.axis0.setdefault(k, None)
        spec.axis1.setdefault(k, None)

    return spec


# =============================================================================
# PART 3.  Applying permutations
# =============================================================================

def _inv(assign: Sequence[int]) -> torch.Tensor:
    """assign[l] = global index of local neuron l  ->  index tensor that
    gathers local neurons into global order."""
    a = np.asarray(assign, dtype=np.int64)
    inv = np.empty_like(a)
    inv[a] = np.arange(len(a))
    return torch.from_numpy(inv)


def permute_state_dict(state_dict: "OrderedDict[str, torch.Tensor]",
                       assignment: Dict[str, Sequence[int]],
                       spec: ResNetPermSpec) -> "OrderedDict[str, torch.Tensor]":
    """Re-index one model's parameters into the global neuron ordering.

    This is a pure relabelling: the permuted network computes exactly the same
    function as the original (see ``verify_permutation_invariance``).
    """
    out = OrderedDict()
    cache: Dict[str, torch.Tensor] = {}
    for g, a in assignment.items():
        cache[g] = _inv(a)

    for k, v in state_dict.items():
        t = v
        g0, g1 = spec.axis0.get(k), spec.axis1.get(k)
        if g0 is not None and t.dim() >= 1 and g0 in cache and t.shape[0] == len(cache[g0]):
            t = t.index_select(0, cache[g0])
        if g1 is not None and t.dim() >= 2 and g1 in cache and t.shape[1] == len(cache[g1]):
            t = t.index_select(1, cache[g1])
        out[k] = t.clone()
    return out


def verify_permutation_invariance(model_factory, state_dict, assignment, spec,
                                  input_shape=(4, 3, 32, 32), tol=1e-4, seed=0) -> Tuple[bool, float]:
    """Sanity check: permuting a model by its own assignment must not change
    what the model computes.  If the permutation groups were wrong (e.g. a
    residual stream was split), this fails loudly."""
    torch.manual_seed(seed)
    x = torch.randn(*input_shape)

    m1 = model_factory()
    m1.load_state_dict(state_dict)
    m1.eval()

    m2 = model_factory()
    m2.load_state_dict(permute_state_dict(state_dict, assignment, spec))
    m2.eval()

    with torch.no_grad():
        y1, y2 = m1(x), m2(x)
    err = (y1 - y2).abs().max().item()
    return err < tol, err


# =============================================================================
# PART 4.  Atom construction + the full merge
# =============================================================================

def _atoms_for_group(state_dict, spec, g, assignment, mode="full",
                     include_bn=True, include_bn_stats=False,
                     include_classifier=True) -> Optional[np.ndarray]:
    """Build the (L, D) matrix of PFNM atoms for one permutation group.

    mode='outgoing' reproduces the paper's top-down formulation (an atom is a
    neuron's vector of outgoing weights).  mode='full' also concatenates the
    incoming weights, which is what makes the method usable when the local
    models have *disjoint* label sets: with disjoint classes the outgoing
    weights into the classifier have disjoint support across models, so the
    matching cost at the top layer is nearly constant and carries no signal.
    """
    parts = []
    L = spec.group_size[g]

    if mode in ("full", "incoming"):
        for k in spec.params_produced_by(g):
            if any(k.endswith(s) for s in _BUFFER_SUFFIXES) and not include_bn_stats:
                continue
            t = state_dict[k]
            if t.dim() == 0 or t.shape[0] != L:
                continue
            if t.dim() == 1 and not include_bn:
                continue
            g1 = spec.axis1.get(k)
            if g1 is not None and t.dim() >= 2 and g1 in assignment:
                t = t.index_select(1, _inv(assignment[g1]))
            parts.append(t.reshape(L, -1).float())

    if mode in ("full", "outgoing"):
        for k in spec.params_consumed_by(g):
            if any(k.endswith(s) for s in _BUFFER_SUFFIXES) and not include_bn_stats:
                continue
            base = k.rsplit(".", 1)[0]
            if not include_classifier and base in ("linear", "fc"):
                continue
            t = state_dict[k]
            if t.dim() < 2 or t.shape[1] != L:
                continue
            g0 = spec.axis0.get(k)
            if g0 is not None and g0 in assignment:
                t = t.index_select(0, _inv(assignment[g0]))
            parts.append(t.transpose(0, 1).reshape(L, -1).float())

    if not parts:
        return None
    return torch.cat(parts, dim=1).detach().cpu().numpy().astype(np.float64)


def _derive_coupled(assignment: Dict[str, List[int]], spec: ResNetPermSpec):
    """Fill in the permutations of residual streams that sit behind a
    parameter-free zero-padding shortcut.

    With ``out = conv_path(x) + pad(x_prev)``, the padded block of channels
    inherits the previous stream's permutation exactly:
        perm_out[offset + i] = offset + perm_prev[i],
    and the remaining (zero-padded) positions are left as identity.
    """
    for g, (src, offset) in spec.coupled_streams.items():
        n_out = spec.group_size[g]
        n_in = spec.group_size[src]
        perm = list(range(n_out))
        if 0 <= offset and offset + n_in <= n_out and src in assignment:
            src_perm = assignment[src]
            for i in range(n_in):
                perm[offset + i] = offset + src_perm[i]
        assignment[g] = perm


def pfnm_merge_state_dicts(
    state_dicts: List["OrderedDict[str, torch.Tensor]"],
    model_factory=None,
    sigma: float = 1.0,
    sigma0: float = 1.0,
    gamma: float = 1.0,
    it: int = 5,
    sweeps: int = 3,
    atom_mode: str = "full",
    allow_growth: bool = False,
    sum_keys: Sequence[str] = ("linear.weight", "linear.bias", "fc.weight", "fc.bias"),
    buffer_plain_mean: bool = True,
    include_classifier_in_atoms: bool = False,
    permute_streams: bool = True,
    verify: bool = True,
    verbose: bool = True,
    seed: int = 0,
) -> Tuple["OrderedDict[str, torch.Tensor]", Dict]:
    """Merge J ResNet state_dicts with PFNM.

    Parameters
    ----------
    state_dicts : list of state_dicts, all with identical shapes.
    model_factory : zero-arg callable returning a fresh model, used only for
        the permutation-invariance self-check.
    sigma, sigma0, gamma : PFNM hyper-parameters (paper defaults are 1.0).
    it : inner leave-one-out iterations inside each group (pfnm.py uses 5).
    sweeps : outer coordinate-descent sweeps over the groups.
    atom_mode : 'full' | 'outgoing' | 'incoming'.  See ``_atoms_for_group``.
    allow_growth : enable PFNM's nonparametric growth.  Keep False to stay
        inside a fixed ResNet-20.
    sum_keys : parameters merged by summation instead of the Gaussian
        posterior mean.  Used for the classifier, whose rows are disjoint
        across specialists, so summing = placing each specialist's rows.
    permute_streams : if False, residual-stream groups are left at identity and
        only block-internal channels are matched.  Safe fallback.

    Returns
    -------
    (merged_state_dict, info)
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
                    f"model {j} has {tuple(sd[k].shape)}. PFNM here assumes identical architectures."
                )

    spec = build_resnet_perm_spec(ref)
    if verbose:
        print("  PFNM permutation groups discovered from the state_dict:")
        print(spec.describe())

    # ---- which groups do we actually solve for? --------------------------
    groups_to_match = [g for g in spec.free_groups]
    if not permute_streams:
        groups_to_match = [g for g in groups_to_match if g not in spec.stream_groups]

    # ---- initialise every permutation to identity ------------------------
    assignments: List[Dict[str, List[int]]] = [
        {g: list(range(n)) for g, n in spec.group_size.items()} for _ in range(J)
    ]

    info = {"groups": OrderedDict(spec.group_size), "shortcut_style": spec.shortcut_style,
            "matched_groups": list(groups_to_match), "sweeps": sweeps,
            "sigma": sigma, "sigma0": sigma0, "gamma": gamma,
            "allow_growth": allow_growth, "atom_mode": atom_mode,
            "skipped_groups": [], "global_sizes": {}}

    # ---- coordinate descent over groups ----------------------------------
    for sweep in range(sweeps):
        n_changed = 0
        for g in groups_to_match:
            atoms = []
            ok = True
            for j in range(J):
                a = _atoms_for_group(state_dicts[j], spec, g, assignments[j], mode=atom_mode,
                                     include_classifier=include_classifier_in_atoms)
                if a is None or a.shape[1] == 0:
                    ok = False
                    break
                atoms.append(a)
            if not ok:
                if sweep == 0:
                    warnings.warn(f"Group '{g}' has no atom features; leaving it unpermuted.")
                    info["skipped_groups"].append(g)
                continue

            assign, _ = match_group(atoms, sigma=sigma, sigma0=sigma0, gamma=gamma,
                                    it=it, allow_growth=allow_growth, rng=rng)

            gsize = max(max(a) + 1 for a in assign)
            info["global_sizes"][g] = gsize
            if gsize != spec.group_size[g]:
                raise RuntimeError(
                    f"PFNM grew group '{g}' from {spec.group_size[g]} to {gsize} neurons. "
                    f"A fixed ResNet-20 cannot represent that. Re-run with allow_growth=False "
                    f"(default) or lower gamma."
                )

            for j in range(J):
                if assign[j] != assignments[j][g]:
                    n_changed += 1
                assignments[j][g] = list(assign[j])

        # streams behind parameter-free shortcuts are derived, not solved
        for j in range(J):
            _derive_coupled(assignments[j], spec)

        if verbose:
            print(f"    sweep {sweep + 1}/{sweeps}: {n_changed} group/model assignments updated")
        if n_changed == 0:
            break

    # ---- self-check -------------------------------------------------------
    if verify and model_factory is not None:
        all_ok = True
        for j in range(J):
            ok, err = verify_permutation_invariance(model_factory, state_dicts[j],
                                                    assignments[j], spec)
            if not ok:
                all_ok = False
                if verbose:
                    print(f"    [!] permutation invariance FAILED for model {j} (max |dy| = {err:.3e})")
        info["permutation_invariance_ok"] = all_ok
        if all_ok and verbose:
            print("    permutation invariance verified for all models (functions unchanged)")
        if not all_ok:
            if permute_streams:
                warnings.warn(
                    "Residual-stream permutations are not function preserving for this "
                    "architecture. Falling back to permuting block-internal channels only."
                )
                return pfnm_merge_state_dicts(
                    state_dicts, model_factory=model_factory, sigma=sigma, sigma0=sigma0,
                    gamma=gamma, it=it, sweeps=sweeps, atom_mode=atom_mode,
                    allow_growth=allow_growth, sum_keys=sum_keys,
                    buffer_plain_mean=buffer_plain_mean,
                    include_classifier_in_atoms=include_classifier_in_atoms,
                    permute_streams=False, verify=verify, verbose=verbose, seed=seed)
            raise RuntimeError("PFNM produced a non-function-preserving permutation.")

    # ---- permute, then aggregate (Prop. 1 / Prop. 4) ----------------------
    permuted = [permute_state_dict(state_dicts[j], assignments[j], spec) for j in range(J)]

    merged = OrderedDict()
    shrink = 1.0 / (1.0 / sigma0 + J / sigma)
    for k, v in ref.items():
        stack = torch.stack([permuted[j][k].float() for j in range(J)])
        if k in sum_keys:
            out = stack.sum(dim=0)
        elif k.endswith("num_batches_tracked"):
            out = stack.max(dim=0).values
        elif buffer_plain_mean and any(k.endswith(s) for s in _BUFFER_SUFFIXES):
            out = stack.mean(dim=0)
        else:
            # theta_hat = (mu0/sigma0 + sum_j v_j/sigma) / (1/sigma0 + J/sigma), mu0 = 0
            out = (stack.sum(dim=0) / sigma) * shrink
        merged[k] = out.to(v.dtype)

    info["assignments"] = assignments
    info["spec"] = spec
    info["shrinkage"] = (J / sigma) * shrink
    return merged, info


# =============================================================================
# PART 5.  BatchNorm re-estimation (forward passes only - no training)
# =============================================================================

@torch.no_grad()
def recalibrate_batchnorm(model, loader, num_batches: int = 100, device: str = "cpu",
                          verbose: bool = True):
    """Re-estimate BN running statistics of a merged model.

    Merging (or averaging) channels invalidates the running mean/var inherited
    from the parents - this is the "variance collapse" that REPAIR (Jordan et
    al., 2023) fixes for interpolated models, and it is usually worth several
    accuracy points.  Only forward passes are used: no gradients, no labels,
    no parameter updates.
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return model
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None          # cumulative moving average
    model.train()
    seen = 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        model(x.to(device))
        seen += 1
        if seen >= num_batches:
            break
    model.eval()
    if verbose:
        print(f"  BatchNorm statistics re-estimated over {seen} batches "
              f"({len(bns)} BN layers, forward passes only)")
    return model
