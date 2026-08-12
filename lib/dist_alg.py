import math
import numpy as np
from scipy.stats import norm
from utils import sgd_grad

# For print logs
RED = "\033[91m"
RESET = "\033[0m"

def _alie_z(n, m):
    s = (n // 2 + 1) - m
    p = (n - m - s) / max(n - m, 1)
    return float(norm.ppf(np.clip(p, 1e-6, 1.0 - 1e-6)))

def byz_atk(i, barrier, models, int_models, X_train, y_train, dp,
            C, K, W, G, H, 
            reg_param, batch_sz, alphas, atk_type, rng, 
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
                barrier.wait()
                models[i] = sum(W[i, j] * int_models[j] for j in list(G.neighbors(i))+[i])
                barrier.wait()
                barrier.wait()
                
        case 'sign_flip':
            flip_scale=kwargs.get('sign_flip_scale',1.0)
            for k in range(K):
                _, g = sgd_grad(Xl, yl, models[i], reg_param, batch_sz, rng)
                int_models[i] = models[i] + flip_scale * alphas[k] * g
                barrier.wait()
                barrier.wait()
                models[i] = sum(W[i, j] * int_models[j] for j in list(G.neighbors(i))+[i])
                barrier.wait()
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
                barrier.wait()
                models[i] = sum(W[i, j] * int_models[j] for j in list(G.neighbors(i))+[i])
                barrier.wait()
                barrier.wait()

        case 'ALIE' | 'IPM':
            H_set = set(int(j) for j in H)
            hon_nbors = [int(j) for j in nbors if int(j) in H_set]
            n_hon = len(hon_nbors)

            alie_z  = _alie_z(len(nbors) + 1, len(nbors) - n_hon + 1)
            ipm_eps = kwargs['ipm_eps']

            for k in range(K):
                barrier.wait()
                if kwargs['abstain'][i]:
                    _, g = sgd_grad(Xl, yl, models[i], reg_param, batch_sz, rng)
                    int_models[i] = models[i] - alphas[k] * g
                else:
                    x_hon = models[hon_nbors].copy()
                    msgs  = int_models[hon_nbors].copy()
                    if atk_type == 'ALIE':
                        int_models[i] = msgs.mean(axis=0) - alie_z * msgs.std(axis=0, ddof=1)
                    else:
                        g_hat = (x_hon - msgs) / alphas[k]
                        int_models[i] = models[i] + ipm_eps * alphas[k] * g_hat.mean(axis=0)
                barrier.wait()
                models[i] = sum(W[i, j] * int_models[j] for j in nbors + [i])
                barrier.wait()
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
                if(sum(W[i, j] for j in U) == 0): return W[i, i] * int_models[i]
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
