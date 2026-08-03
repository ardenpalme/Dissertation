'''
TODO: understand byzantine attack implementation and aggregator implementations
'''
import math
import warnings
import numpy as np
from scipy.stats import norm
from preprocessor import MatrixSummaryFeatures

from utils import sgd_grad

def _alie_z(n, m):
    """Baruch et al. (2019) perturbation coefficient.
 
    s = floor(n/2 + 1) - m is the number of honest messages that must be
    displaced before a median-type aggregator moves; z is the largest
    deviation, in honest standard deviations, that still leaves the Byzantine
    message inside the bulk of the n - m honest messages.
    """
    s = (n // 2 + 1) - m
    p = (n - m - s) / max(n - m, 1)
    return float(norm.ppf(np.clip(p, 1e-6, 1.0 - 1e-6)))

# -----------------------------------------------------------------------------
# Threat model IV helpers: the adversary replicates the victim's own detector
# call.  MatrixSummaryFeatures.transform() reads the globals `models` and
# `alphas`, and uses self.node_id as the anchor, so instantiating it with the
# victim's id reproduces exactly the feature vector the victim would compute.
# -----------------------------------------------------------------------------
 
def _victim_feat(v, knowledge, alphas, X_train, y_train, dp, Xl, yl, rng, n_proxy=2000):
    """Feature extractor as victim v would build it.
 
    knowledge='full'  : adversary knows the victim's local shard (upper bound
                        on adversary strength; omniscient-but-not-omnipotent).
    knowledge='proxy' : adversary substitutes its own shard, which it always
                        has under threat model III onwards.  Under Dirichlet
                        heterogeneity these shards differ, so evasion degrades
                        gracefully and the gap between the two settings
                        measures how much the detector's privacy of local data
                        is actually worth.
    """
    if knowledge == 'full':
        Xv, yv = X_train[dp[v]], y_train[dp[v]]
    else: # knowledge = 'proxy' (available at threat models III)
        Xv, yv = Xl, yl
    if n_proxy and len(yv) > n_proxy:
        sub = rng.choice(len(yv), size=n_proxy, replace=False)
        Xv, yv = Xv[sub], yv[sub]
    return MatrixSummaryFeatures(v, Xv, yv, alphas)
 
 
def _byz_score(msg, victims, models, feats, g_est, k, pre_proc, best_est):
    """Worst-case detector score of a single broadcast message.
 
    The message is broadcast, so it must clear every honest neighbour's
    detector at once; the binding constraint is the maximum over victims.
    """
    worst = 0.0
    for v, fe in zip(victims, feats):
        fe.set_context(k, models[v], g_est)
        Z = fe.transform(msg[None, ...])
        p = float(best_est.predict_proba(pre_proc.transform(Z))[0, 1])
        worst = max(worst, p)
    return worst
 
def _clip(v, tau):
    """Self-centred projection P_tau applied to a displacement (definition 2)."""
    n = np.linalg.norm(v)
    return v if n <= tau or n == 0.0 else (tau / n) * v

 
def _best_evasive_msg(mu, u_atk, u_hon, tau_k, rho_grid, eta_grid,
                      tau_C, margin, victims, feats, g_est, k,
                      models,
                      pre_proc, best_est):
    """Strongest admitted message, searching direction *and* magnitude.
 
    Searching over magnitude alone would be close to searching a constant.
    Write the reconstructed pseudo-gradient as G = (x_i - msg) / alpha_k and
    the message as mu + eta*d.  Of the seven features:
 
      margin      _shift_scores normalises both operands  -> scale-free
      stable_rank ratio of squared singular values        -> scale-free
      abs_cos_g   a cosine                                -> scale-free
      entropy     depends on argmax(X_l @ msg)            -> saturates in eta
      acc         depends on argmax(X_l @ msg)            -> saturates in eta
      ce_diff     softmax CE, grows with the logit scale  -> responds to eta
 
    So the score is governed by the *direction* of the displacement, and only
    ce_diff resists magnitude at all.  The adversary's real search space is
    therefore the direction: rho blends the ascent direction u_atk towards
    u_hon, an observed honest displacement, buying camouflage at the cost of
    alignment.
 
    Magnitude is bounded by the receiver's clipping radius rather than by the
    detector, which is precisely the situation section 4.3.2 describes: with a
    scale-free detector, corollary 10 is the only remaining control on message
    length.  The objective reflects this — the payoff is the component of the
    *clipped* displacement along the ascent direction, since displacement the
    receiver clips away and displacement orthogonal to u_atk both do no damage.
 
    Returns (msg, damage, profile); profile rows are (rho, eta, damage, score)
    and are the raw material for the attack-strength vs detection figure.
    """
    profile = []
    best_msg, best_J = mu.copy(), 0.0
    for rho in rho_grid:
        d = (1.0 - rho) * u_atk + rho * u_hon
        nd = np.linalg.norm(d)
        if nd == 0.0:
            continue
        d = d / nd # normalized candidate attack direction
        for eta in eta_grid:
            msg = mu + eta * d # control deviation from honest gradient average
            # Clipping is centred on the receiver's x_i, which differs per
            # victim; mu is the adversary's best available proxy for it.
            J = float((_clip(msg - mu, tau_k) * u_atk).sum())
            p = _byz_score(msg, victims, models, feats, g_est, k, pre_proc, best_est)
            profile.append((float(rho), float(eta), J, p))
            if p <= tau_C - margin and J > best_J:
                best_msg, best_J = msg, J
    return best_msg, best_J, profile

def byz_atk(i, barrier, models, int_models, X_train, y_train, dp,
            C, K, W, G, H, 
            reg_param, batch_sz, alphas, taus, atk_type, rng, 
            atk_log=None,
            **kwargs):
    Xl, yl = X_train[dp[i]], y_train[dp[i]]
    nbors = list(G.neighbors(i))

    match atk_type:
        case 'label_flip':
            yl_perm = (yl + rng.integers(1,C)) % C
            for k in range(K):
                _, g = sgd_grad(Xl, yl_perm, models[i], reg_param, batch_sz, rng)
                int_models[i] = models[i] - alphas[k] * g
                barrier.wait()
                models[i] = sum(W[i, j] * int_models[j] for j in list(G.neighbors(i))+[i])
                barrier.wait()
                
        case 'sign_flip':
            flip_scale=1.0
            for k in range(K):
                _, g = sgd_grad(Xl, yl, models[i], reg_param, batch_sz, rng)
                int_models[i] = models[i] + flip_scale * alphas[k] * g
                barrier.wait()
                models[i] = sum(W[i, j] * int_models[j] for j in list(G.neighbors(i))+[i])
                barrier.wait()
                
        case 'gaussian':
            sigma=1.0
            around_honest=True
            for k in range(K):
                if around_honest:
                    _, g = sgd_grad(Xl, yl, models[i], reg_param, batch_sz, rng)
                    base = models[i] - alphas[k] * g
                else:
                    base = 0.0
                int_models[i] = base + sigma * rng.standard_normal(models[i].shape)
                barrier.wait()
                models[i] = sum(W[i, j] * int_models[j] for j in list(G.neighbors(i))+[i])
                barrier.wait()

        # ------------------------------------------------------------------
        # Threat model IV: adaptive, one-round-delayed observation of honest
        # neighbours.  Two race-free observation windows exist per iteration:
        #   (a) between barrier 2 of k-1 and barrier 1 of k, `models` is stable
        #       (honest workers only write `int_models` in that window);
        #   (b) between barrier 1 and barrier 2 of k, `int_models` is stable
        #       (honest workers only write `models` in that window).
        # Snapshotting (a) then (b) yields the honest messages x_j^{k+1/2} and
        # the honest pseudo-gradients g_j = (x_j^k - x_j^{k+1/2}) / alpha_k,
        # both usable at iteration k+1.  Reading either array outside its
        # window is a data race and destroys run-to-run reproducibility.
        # ------------------------------------------------------------------
        case 'ALIE' | 'IPM':
            H_set = set(int(j) for j in H)
            hon_nbors = [int(j) for j in nbors if int(j) in H_set]
            if not hon_nbors:
                warnings.warn(f"node {i}: no honest neighbour, {atk_type} "
                              f"degenerates to honest behaviour")

            if kwargs['alie_z'] is None: alie_z = _alie_z(kwargs['num_nodes'], kwargs['b'])
            else: alie_z = kwargs['alie_z']
            ipm_eps = kwargs['ipm_eps']
            
            prev_msgs = None   # (|hon_nbors|, d+1, C) honest messages at k-1
            prev_grads = None  # (|hon_nbors|, d+1, C) honest pseudo-grads at k-1

            for k in range(K):
                if prev_msgs is None or prev_grads is None:
                    # k = 0, or no honest neighbour to observe: send an honest
                    # update so the attack has a well-defined warm start.
                    # Both snapshots are written together under `if hon_nbors`,
                    # so this branch also covers the degenerate case of a
                    # Byzantine node whose neighbourhood contains no honest
                    # node, in which case it behaves honestly for all K rounds.
                    _, g = sgd_grad(Xl, yl, models[i], reg_param, batch_sz, rng)
                    int_models[i] = models[i] - alphas[k] * g
 
                elif atk_type == 'ALIE':
                    # Coordinate-wise mean/std of the honest messages; sit z
                    # standard deviations below the mean, i.e. inside the
                    # honest population but biased against it.
                    int_models[i] = prev_msgs.mean(axis=0) - alie_z * prev_msgs.std(axis=0)
 
                elif atk_type == 'IPM':  
                    # Send an ascent step of size eps along the mean honest
                    # pseudo-gradient, so that the inner product between the
                    # aggregate and the true descent direction is negative.
                    # Anchored at models[i]: the Byzantine node mixes honestly,
                    # so x_i tracks the consensus and the receiver's
                    # reconstruction (x_recv - msg)/alpha_k is ~ -eps * gbar.
                    int_models[i] = models[i] + ipm_eps * alphas[k] * prev_grads.mean(axis=0)

                # window (a): `models` still holds x^k for every node
                x_hon = models[hon_nbors].copy() if hon_nbors else None
 
                barrier.wait()
 
                # window (b): `int_models` holds x^{k+1/2} for every node
                if hon_nbors:
                    prev_msgs = int_models[hon_nbors].copy()
                    prev_grads = (prev_msgs - x_hon) / alphas[k]
 
                models[i] = sum(W[i, j] * int_models[j] for j in nbors + [i])
                barrier.wait()

        case 'EVADE':
            H_set = set(int(j) for j in H)
            hon_nbors = [int(j) for j in nbors if int(j) in H_set]
            if not hon_nbors:
                warnings.warn(f"node {i}: no honest neighbour, {atk_type} "
                              f"degenerates to honest behaviour")

            C_objs, know, feats, last_msg = {}, '', [], None
            margin, n_eta, n_rho, rho_max, reach, atk_freq = 0,0,0,0,0,0
            if atk_type == 'EVADE':
                if kwargs['classifier'] is None:
                    raise ValueError("'evade' needs kwargs['classifier'] containing pre_proc, best_est, C_tau")

                C_objs = kwargs['classifier']
                know = kwargs['atk_knowledge'] # proxy
                margin = kwargs['atk_margin']  # 0.05
                n_eta = kwargs['atk_eta_grid'] # 4
                n_rho = kwargs['atk_rho_grid'] # 4
                rho_max = kwargs['atk_rho_max'] # 0.8
                reach = kwargs['atk_reach']    # 2.0
                atk_freq = kwargs['atk_freq']    # attack every _ iters
                feats = [_victim_feat(v, know, alphas, X_train, y_train, dp, Xl, yl, rng) for v in hon_nbors]
                last_msg = None

            prev_msgs = None   # (|hon_nbors|, d+1, C) honest messages at k-1
            prev_grads = None  # (|hon_nbors|, d+1, C) honest pseudo-grads at k-1

            for k in range(K):
                if prev_msgs is None or prev_grads is None:
                    # k = 0, or no honest neighbour to observe: send an honest
                    # update so the attack has a well-defined warm start.
                    # Both snapshots are written together under `if hon_nbors`,
                    # so this branch also covers the degenerate case of a
                    # Byzantine node whose neighbourhood contains no honest
                    # node, in which case it behaves honestly for all K rounds.
                    _, g = sgd_grad(Xl, yl, models[i], reg_param, batch_sz, rng)
                    int_models[i] = models[i] - alphas[k] * g

                elif atk_type == 'EVADE':  # 'evade' — section 4.3.2
                    # Anchor at the honest mean message, so the message is
                    # honest-looking by construction, then search over the
                    # displacement direction and magnitude.
                    mu = prev_msgs.mean(axis=0)
                    gbar = prev_grads.mean(axis=0)
 
                    # Ascent direction: honest displacement is -alpha*g, so
                    # +gbar is the direction that undoes honest progress.
                    u_atk = gbar / (np.linalg.norm(gbar) + 1e-12)
 

                    # Camouflage direction: the principal axis of the honest
                    # displacement cloud.  Dv has zero mean by construction
                    # (mu is the mean of prev_msgs), so the cloud has no mean
                    # direction, only a covariance; its top principal axis is
                    # where honest messages are most widely scattered and
                    # therefore where a deviation is least distinguishable
                    # from honest disagreement.  Since x_j - mu ~ -alpha(g_j -
                    # gbar), this axis is the inter-node gradient
                    # disagreement direction, i.e. the heterogeneity zeta the
                    # detector must tolerate to keep its false positive rate
                    # down.

                    '''
                    What carries the camouflage is its principal axis: the direction in which honest messages
                    are most widely scattered, and therefore where a deviation is least separable from ordinary 
                    honest disagreement. Now computed by SVD of the flattened displacements, with the random fallback 
                    reserved for the degenerate single-neighbour case where Dv is identically zero.
                    '''
                    Dv = prev_msgs - mu
                    Df = Dv.reshape(Dv.shape[0], -1)
                    if Dv.shape[0] > 1 and np.linalg.norm(Df) > 1e-12:
                        _, _, Vt = np.linalg.svd(Df, full_matrices=False)
                        u_hon = Vt[0].reshape(mu.shape)
                    else:
                        u_hon = rng.standard_normal(mu.shape)
                    u_hon = u_hon / (np.linalg.norm(u_hon) + 1e-12)
 
                    # Magnitude is bounded by the receiver's clipping radius,
                    # not by the detector: reach > 1 only exposes the clipped
                    # region on the profile, where extra length is pure waste.
                    tau_k = float(taus[k])
 
                    if k % atk_freq == 0 or last_msg is None:
                        rho_grid = np.linspace(0.0, rho_max, n_rho)
                        eta_grid = np.linspace(0.0, reach * tau_k, n_eta + 1)[1:]
                        msg, J, profile = _best_evasive_msg(
                            mu, u_atk, u_hon, tau_k, rho_grid, eta_grid,
                            C_objs['C_tau'], margin, hon_nbors, feats, gbar, k,
                            models,
                            C_objs['pre_proc'], C_objs['best_est'])
                        last_msg = msg
                        if atk_log is not None:
                            atk_log.append({'node': i, 'k': k,
                                            'damage': J,
                                            'tau_k': tau_k,
                                            'displacement': float( np.linalg.norm(msg - mu)),
                                            'admitted': bool(J > 0.0),
                                            'profile': profile})
                    else:
                        msg = last_msg
 
                    # J = 0 means every candidate was flagged: the fallback is
                    # the honest anchor mu, admitted but harmless.  Those
                    # rounds are the detector actually working under threat
                    # model IV — count them, they are the result.
                    int_models[i] = msg

                # window (a): `models` still holds x^k for every node
                x_hon = models[hon_nbors].copy() if hon_nbors else None
 
                barrier.wait()
 
                # window (b): `int_models` holds x^{k+1/2} for every node
                if hon_nbors:
                    prev_msgs = int_models[hon_nbors].copy()
                    prev_grads = (prev_msgs - x_hon) / alphas[k]
 
                models[i] = sum(W[i, j] * int_models[j] for j in nbors + [i])
                barrier.wait()

        case _:
            raise ValueError(f"Unknown attack type: {atk_type}")

def aggregate(i, int_models, W, G, B, H, agg_rule, q_i=None, tau_scc=0.3, q_trim=None): 
    nbrs = list(G.neighbors(i))
    if q_i is None: q_i = math.ceil(np.array([sum(1 for j in list(G.neighbors(i)) if j in B) for i in H]).mean())
    if q_trim is None: q_trim = math.ceil(np.array([sum(1 for j in list(G.neighbors(i)) if j in B) for i in H]).mean())
    
    match agg_rule:
        case "IOS":
            trusted = set(G.neighbors(i)) | {i}
            def wavg(U):
                return sum(W[i, j] * int_models[j] for j in U) / sum(W[i, j] for j in U)
            for _ in range(q_i):
                x_avg = wavg(trusted)
                cands = np.array([j for j in trusted if j != i])
                if cands.size == 0: break
                d = np.linalg.norm(int_models[cands] - x_avg, axis=(1, 2))
                trusted.discard(int(cands[d.argmax()]))
            return wavg(trusted)
 
        case "SCC":
            def _clip(x, tau):
                n = np.linalg.norm(x)
                return x if n <= tau or n == 0.0 else (tau / n) * x
            return int_models[i] + sum(
                W[i, j] * _clip(int_models[j] - int_models[i], tau_scc)
                for j in nbrs)
 
        case "TriMean":
            S = np.stack([int_models[j] for j in nbrs])
            m = S.shape[0]
            q_trim = min(q_trim, max(0, (m - 1) // 2))
            Ssort = np.sort(S, axis=0)
            base = Ssort[q_trim:m - q_trim].mean(0) if q_trim else Ssort.mean(0)
            r = 1.0 / (m + 1 - 2 * q_trim)
            return (1.0 - r) * base + r * int_models[i]
 
        case "CooMed":
            S = np.stack([int_models[j] for j in nbrs])
            return (1.0 - W[i,i]) * np.median(S, axis=0) + W[i,i] * int_models[i]
