"""
Pair-state CRF kernels — state-count-agnostic forward/Viterbi/marginal
recursion with a pair switch matrix. Transition cost is -c*nsw, nsw in
{0,1,2}: a two-chromosome switch costs exp(-2c) = exp(-c)^2, i.e. two
INDEPENDENT chromosome switches (matches the generative sim). No hard ban —
simultaneous switches are rare, not illegal.

Extracted verbatim from train_diploid.py (2026-07: original), pulled out into
this shared module (2026-09, experiments/simulator-indels/TRAINING_PLAN.md §2)
so both the existing diploid-affinity model and the new indel-aware model
(train_diploid_indel.py) import the same kernels rather than duplicating them.
These functions take no raw per-cell features at all — only per-timestep
emission/transition scores — so they are completely independent of what the
input feature width or semantics look like.

Only [B,P,P] is materialized per timestep (never [B,T,P,P]).
"""

import torch


@torch.jit.script
def _dcrf_nll(emis: torch.Tensor, c: torch.Tensor, nsw: torch.Tensor,
              stay_bonus: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
    """Pair-state CRF NLL. emis [B,T,P], c [B,T], nsw/stay [P,P], tags [B,T]."""
    B, T, P = emis.shape
    # fp32 partition (see train_haploid._crf_nll): the sequential logsumexp
    # accumulation needs fp32 precision; encoder stays bf16, no matmuls here.
    emis = emis.float()
    c = c.float()
    nsw = nsw.float()
    stay_bonus = stay_bonus.float()
    stay_mask = (nsw == 0).float()
    a = emis[:, 0]
    for t in range(1, T):
        tr_t = -c[:, t, None, None] * nsw[None] + stay_bonus * stay_mask[None]
        a = emis[:, t] + torch.logsumexp(a.unsqueeze(2) + tr_t, dim=1)
    log_Z = torch.logsumexp(a, dim=1)

    bi = torch.arange(B, device=emis.device)
    t_idx = torch.arange(T, device=emis.device)
    emis_score = emis[bi.unsqueeze(1), t_idx.unsqueeze(0), tags].sum(1)
    # transition score along the true path: -c*nsw[prev,next] + stay where equal
    prev = tags[:, :-1]
    nxt = tags[:, 1:]
    nsw_path = nsw[prev, nxt]                                   # [B,T-1]
    stay_path = (nsw_path == 0).float()
    tr_score = (-c[:, 1:] * nsw_path + stay_bonus * stay_path).sum(1)
    return (log_Z - emis_score - tr_score).mean()


@torch.jit.script
def _dcrf_viterbi(emis: torch.Tensor, c: torch.Tensor, nsw: torch.Tensor,
                  stay_bonus: torch.Tensor) -> torch.Tensor:
    """Pair-state Viterbi. emis [B,T,P] -> [B,T] pair indices."""
    B, T, P = emis.shape
    stay_mask = (nsw == 0).float()
    delta = emis[:, 0]
    bp = torch.zeros(T - 1, B, P, dtype=torch.long, device=emis.device)
    for t in range(1, T):
        tr_t = -c[:, t, None, None] * nsw[None] + stay_bonus * stay_mask[None]
        sc = delta.unsqueeze(2) + tr_t
        best, idx = sc.max(dim=1)
        delta = emis[:, t] + best
        bp[t - 1] = idx
    path = torch.zeros(B, T, dtype=torch.long, device=emis.device)
    path[:, T - 1] = delta.argmax(dim=1)
    for t in range(T - 2, -1, -1):
        path[:, t] = bp[t].gather(1, path[:, t + 1].unsqueeze(1)).squeeze(1)
    return path


def _dcrf_marginal(emis: torch.Tensor, c: torch.Tensor, nsw: torch.Tensor,
                   stay_bonus: torch.Tensor) -> torch.Tensor:
    """Posterior max-marginal (forward-backward) decode. emis [B,T,P] -> [B,T].

    Viterbi returns the single highest-scoring JOINT path; this returns the
    per-site argmax of the posterior marginal P(state_t | x), which maximises
    expected PER-SITE accuracy -- the quantity we actually report. Same forward
    recursion as the partition (_dcrf_nll) plus a symmetric backward pass; the
    transition into time t uses c[:,t], matching the forward convention."""
    emis = emis.float()
    c = c.float()
    nsw = nsw.float()
    stay_bonus = stay_bonus.float()
    stay_mask = (nsw == 0).float()
    B, T, P = emis.shape

    alpha = torch.empty(B, T, P, device=emis.device)
    alpha[:, 0] = emis[:, 0]
    for t in range(1, T):
        tr_t = -c[:, t, None, None] * nsw[None] + stay_bonus * stay_mask[None]
        alpha[:, t] = emis[:, t] + torch.logsumexp(alpha[:, t - 1].unsqueeze(2) + tr_t, dim=1)

    beta = torch.zeros(B, P, device=emis.device)
    pred = torch.empty(B, T, dtype=torch.long, device=emis.device)
    pred[:, T - 1] = (alpha[:, T - 1] + beta).argmax(dim=1)
    for t in range(T - 2, -1, -1):
        tr_n = -c[:, t + 1, None, None] * nsw[None] + stay_bonus * stay_mask[None]  # [B,p=t,q=t+1]
        msg = emis[:, t + 1] + beta                                                 # [B,P] over q
        beta = torch.logsumexp(tr_n + msg.unsqueeze(1), dim=2)                      # [B,P] over p=t
        pred[:, t] = (alpha[:, t] + beta).argmax(dim=1)
    return pred


def _dcrf_viterbi_factored(emis, c, nsw, stay_bonus, pi, pj):
    """O(T·(P+K)) pair-state Viterbi — bit-identical to _dcrf_viterbi but it exploits
    the factored transition (-c·nsw, nsw∈{0,1,2}=per-chromosome switches): the max over
    P prev-states reduces to per-FOUNDER maxima. The step from p→q=(i,j) is either a
    stay (p=q), one switch (p shares founder i or j → use best[i]/best[j]), or two
    switches (any p → global max). **Faster on CPU (~4.6× at P=325), SLOWER on GPU**
    (it trades one fused [B,P,P] kernel for several launch-bound ops) — use for CPU
    whole-genome decode. emis [B,T,P], c [B,T] -> [B,T] pair indices."""
    emis = emis.float(); c = c.float(); stay = stay_bonus.float()
    B, T, P = emis.shape
    dev = emis.device
    K = int(max(pi.max(), pj.max())) + 1
    piB, pjB = pi.expand(B, P), pj.expand(B, P)
    idxP = torch.arange(P, device=dev).expand(B, P)
    catidx = torch.cat([piB, pjB], 1)
    bp = torch.empty(T, B, P, dtype=torch.long, device=dev)
    delta = emis[:, 0].clone()
    bp[0] = idxP
    for t in range(1, T):
        best = torch.full((B, K), float("-inf"), device=dev)
        best.scatter_reduce_(1, catidx, torch.cat([delta, delta], 1),
                             reduce="amax", include_self=True)
        mi, mj = best.gather(1, piB), best.gather(1, pjB)
        # argbest per founder = lowest pair index achieving best (matches torch.max ties)
        cand_i = torch.where(delta == mi, idxP, torch.full_like(idxP, P))
        cand_j = torch.where(delta == mj, idxP, torch.full_like(idxP, P))
        argbest = torch.full((B, K), P, dtype=torch.long, device=dev)
        argbest.scatter_reduce_(1, piB, cand_i, reduce="amin", include_self=True)
        argbest.scatter_reduce_(1, pjB, cand_j, reduce="amin", include_self=True)
        use_i = mi >= mj
        m1 = torch.where(use_i, mi, mj)                                     # one-switch value
        bp1 = torch.where(use_i, argbest.gather(1, piB), argbest.gather(1, pjB))
        gmax, argg = delta.max(1)                                          # two-switch
        ct = c[:, t, None]
        vals = torch.stack([delta + stay, -ct + m1,
                            (-2 * ct + gmax[:, None]).expand(B, P)])        # [3,B,P]
        bps = torch.stack([idxP, bp1, argg[:, None].expand(B, P)])
        bi = vals.argmax(0)
        delta = emis[:, t] + vals.gather(0, bi[None])[0]
        bp[t] = bps.gather(0, bi[None])[0]
    path = torch.empty(B, T, dtype=torch.long, device=dev)
    path[:, T - 1] = delta.argmax(1)
    for t in range(T - 2, -1, -1):
        path[:, t] = bp[t + 1].gather(1, path[:, t + 1:t + 2]).squeeze(1)
    return path


def build_pair_tables(K):
    """Unordered founder pairs over K states. Returns:
      pi, pj      [P]      sorted member indices of each pair (i<=j)
      pair_table  [K,K]    (a,b) -> pair index (order-insensitive)
      nsw_pair    [P,P]    min #chromosome switches between pairs (0,1,2)
    """
    pairs = [(i, j) for i in range(K) for j in range(i, K)]
    P = len(pairs)
    idx = {p: k for k, p in enumerate(pairs)}
    pi = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    pj = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    pair_table = torch.zeros(K, K, dtype=torch.long)
    for a in range(K):
        for b in range(K):
            pair_table[a, b] = idx[(min(a, b), max(a, b))]

    nsw = torch.zeros(P, P, dtype=torch.float32)
    for p, (a, b) in enumerate(pairs):
        for q, (c, d) in enumerate(pairs):
            s = min((a != c) + (b != d), (a != d) + (b != c))
            nsw[p, q] = float(s)
    return pi, pj, pair_table, nsw
