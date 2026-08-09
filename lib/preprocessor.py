import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = [
    'cos_g', 'log_norm_ratio', 'stable_rank', 'margin',
    'shift_off_diag', 'shift_conc', 'entropy', 'ce_diff',
    'acc', 'cos_dev_g', 'log_dev_norm', 'cos_dev_scale',
    'cos_gbar', 'ce_diff_nbr', 'acc_dev', 'log_alpha', 'k_frac',
]

HEAVY_FEATURES = ['log_norm_ratio', 'ce_diff', 'log_dev_norm', 'ce_diff_nbr']

class MatrixSummaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, node_id, X_local, y_local, alphas):
        self.X_local = X_local # (n_loc, d) local (augmented) features
        self.y_local = y_local # (n_loc,)  local labels
        self.node_id = node_id
        self.alphas = alphas

    def set_context(self, k, theta_ref, g):
        self.theta_ref = np.asarray(theta_ref)  # (d,C) current model of node i
        self.g = g # (d,C) current local stochastic gradient
        self.iter = k
        return self

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _softmax_ce(logits, y):
        z = logits - logits.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -log_p[np.arange(len(y)), y].mean()
        
    @staticmethod
    def _shift_scores(theta_i, theta_j, eps=1e-12):
        A = theta_i.T @ theta_j  # (C, C)
        A /= (np.linalg.norm(theta_i, axis=0)[:, None] * np.linalg.norm(theta_j, axis=0)[None, :] + eps)
        C = A.shape[0]
        return np.array([np.mean(np.diag(np.roll(A, -s, axis=1))) for s in range(C)]) # (C,)

    def transform(self, X):
        m, d, C = X.shape # stacked int models of the m neighbours
        eps = 1e-12
 
        th_i = self.theta_ref
        gi = self.g.ravel()
        gn = np.linalg.norm(gi) + eps
        Xl, yl = self.X_local, self.y_local
        alpha_k = float(self.alphas[self.iter])
        K = len(self.alphas)

        Xf = X.reshape(m, -1) # (m, D) flattened neighbour models
        D3 = th_i[None, ...] - X # (m, d, C) raw disagreement
        D = D3.reshape(m, -1) # (m, D)
        Dn = np.linalg.norm(D, axis=1)

        # Features relative to models[i]
        cos_g = (D @ gi) / (Dn * gn + eps)

        log_norm_ratio = np.log(Dn / (alpha_k * gn) + eps)
 
        sv = np.linalg.svd(D3, compute_uv=False)
        stable_rank = (sv ** 2).sum(axis=1) / (sv[:, 0] ** 2 + eps)

        M = (Xl.T @ np.eye(C)[yl]) / len(yl)
        sig = np.array([self._shift_scores(M, D3[j]) for j in range(m)])
        margin = sig.max(axis=1) - sig[:, 0]
        shift_off_diag = (sig.argmax(axis=1) != 0).astype(float)
        sig_pos = np.clip(sig, 0.0, None)
        shift_conc = sig_pos.max(axis=1) / (sig_pos.sum(axis=1) + eps)

        logits_i = Xl @ th_i
        pred_i = logits_i.argmax(axis=1)
        ce_i = self._softmax_ce(logits_i, yl)
 
        ent = np.empty(m)
        ce_abs = np.empty(m)
        acc = np.empty(m)
        for j in range(m):
            logits_j = Xl @ X[j]
            pred_j = logits_j.argmax(axis=1)
            p = np.bincount((pred_j - pred_i) % C, minlength=C) / len(pred_i) + eps
            ent[j] = -(p * np.log(p)).sum()
            ce_abs[j] = self._softmax_ce(logits_j, yl)
            acc[j] = (pred_j == yl).mean()
        ce_diff = ce_abs - ce_i

        # Leave-one-out neighbourhood-relative features
        if m >= 3:
            dev = np.empty_like(Xf)  # (m,D)
            scl = np.empty_like(Xf)  # (m,D)
            gbar = np.empty_like(D)  # (m,D)
            ce_med = np.empty(m)
            acc_med = np.empty(m)
            for j in range(m):
                O = np.delete(Xf, j, axis=0)
                mu = np.median(O, axis=0)
                dev[j] = Xf[j] - mu
                scl[j] = 1.4826 * np.median(np.abs(O - mu), axis=0) + eps
                gbar[j] = np.median(np.delete(D, j, axis=0), axis=0)
                ce_med[j] = np.median(np.delete(ce_abs, j))
                acc_med[j] = np.median(np.delete(acc, j))

            dn = np.linalg.norm(dev, axis=1)
 
            cos_dev_g = (dev @ gi) / (dn * gn + eps)
            log_dev_norm = np.log(dn / (np.median(dn) + eps) + eps)
            cos_dev_scale = ((dev * scl).sum(1) / (dn * np.linalg.norm(scl, axis=1) + eps))
            cos_gbar = ((D * gbar).sum(1) / (Dn * np.linalg.norm(gbar, axis=1) + eps))
 
            ce_diff_nbr = ce_abs - ce_med
            acc_dev = acc - acc_med
        else:
            zeros = np.zeros(m)
            cos_dev_g = zeros
            log_dev_norm = zeros
            cos_dev_scale = zeros
            cos_gbar = zeros
            ce_diff_nbr = zeros
            acc_dev = zeros

        # Context Features
        log_alpha = np.full(m, np.log(alpha_k + eps))
        k_frac = np.full(m, self.iter / max(K - 1, 1))
 
        feats = np.column_stack([
            cos_g, log_norm_ratio, stable_rank,
            margin, shift_off_diag, shift_conc,
            ent, ce_diff, acc,
            cos_dev_g, log_dev_norm, cos_dev_scale, cos_gbar,
            ce_diff_nbr, acc_dev,
            log_alpha, k_frac,
        ])

        assert feats.shape[1] == len(FEATURE_NAMES) 
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

