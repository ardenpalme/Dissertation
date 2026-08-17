import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = [
    'cos_g', 
    'log_norm_ratio', 
    'margin',
    'stable_rank', 
    'acc',
    'cos_gbar',
    'log_dev_norm', 
    'ce_diff_nbr', 
    'log_alpha', 
]

HEAVY_FEATURES = ['log_norm_ratio', 'log_dev_norm', 'ce_diff_nbr']

class FeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, node_id, X_local, y_local, alphas, reg_param):
        self.X_local = X_local # (n_loc, d) local (augmented) features
        self.y_local = y_local # (n_loc,)  local labels
        self.node_id = node_id
        self.alphas = alphas
        self.reg_param = reg_param

    def set_context(self, k, theta_ref, g):
        self.theta_ref = np.asarray(theta_ref)  # (d,C) current model of node i
        self.g = g # (d,C) current local stochastic gradient
        self.iter = k
        return self

    def fit(self, X, y=None):
        return self

    def _precompute(self, C):
        if getattr(self, '_pre_C', None) == C: return
        yl = self.y_local
        cnt = np.bincount(yl, minlength=C).astype(float)
        self._cnt = cnt
        self._sup = np.flatnonzero(cnt > 0)
        self._n_sup = len(self._sup)
        self._w_bal = 1.0 / (self._n_sup * cnt[yl])
        self._pre_C = C

    def _softmax_ce(self, logits, y, x=None, w=None):
        z = logits - logits.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        nll = -log_p[np.arange(len(y)), y]
        ce = float(nll.mean()) if w is None else float(nll @ w)
        if x is not None:
            ce += 0.5 * self.reg_param * float((x ** 2).sum())
        return ce

    def _shift_profile(self, M, U, eps=1e-12):
        A = M.T @ U 
        A = A / (np.linalg.norm(M, axis=0)[:, None]
                 * np.linalg.norm(U, axis=0)[None, :] + eps)
        C = A.shape[0]
        sup = self._sup
        return np.array([-np.mean(np.diag(np.roll(A, -s, axis=1))[sup]) for s in range(C)]) 

    def transform(self, X):
        m, d, C = X.shape
        eps = 1e-12

        self._precompute(C)
        w_bal = self._w_bal
 
        th_i = self.theta_ref
        gi = self.g.ravel()
        gn = np.linalg.norm(gi) + eps
        Xl, yl = self.X_local, self.y_local
        alpha_k = float(self.alphas[self.iter])

        Xf = X.reshape(m, -1)    # (m, d*C) flattened neighbour models
        D3 = th_i[None, ...] - X # (m, d, C) 
        D = D3.reshape(m, -1)    # (m, d*C)
        Dn = np.linalg.norm(D, axis=1)

        # Features relative to models[i]
        cos_g = (D @ gi) / (Dn * gn + eps)

        log_norm_ratio = np.log(Dn / (alpha_k * gn) + eps)
 
        sv = np.linalg.svd(D3, compute_uv=False)
        stable_rank = (sv ** 2).sum(axis=1) / (sv[:, 0] ** 2 + eps)

        M = (Xl.T @ np.eye(C)[yl]) / len(yl)
        rho = np.array([self._shift_profile(M, D3[j]) for j in range(m)])   # (m, C)
        margin = rho.max(axis=1) - rho[:, 0]          

        ce_abs = np.empty(m)
        acc = np.empty(m)
        for j in range(m):
            logits_j = Xl @ X[j]
            ce_abs[j] = self._softmax_ce(logits_j, yl, X[j], w_bal) 

        # Leave-one-out neighbourhood-relative features
        if m >= 3:
            dev = np.empty_like(Xf)  # (m, d*C)
            gbar = np.empty_like(D)  # (m, d*C)
            ce_med = np.empty(m)
            for j in range(m):
                O = np.delete(Xf, j, axis=0)
                mu = np.median(O, axis=0)
                dev[j] = Xf[j] - mu
                gbar[j] = np.median(np.delete(D, j, axis=0), axis=0)
                ce_med[j] = np.median(np.delete(ce_abs, j))

            log_dev_norm = np.log(np.linalg.norm(dev, axis=1) / (np.median(np.linalg.norm(dev, axis=1)) + eps) + eps)
            cos_gbar = ((D * gbar).sum(1) / (Dn * np.linalg.norm(gbar, axis=1) + eps))
 
            ce_diff_nbr = ce_abs - ce_med
        else:
            zeros = np.zeros(m)
            log_dev_norm = zeros
            cos_gbar = zeros
            ce_diff_nbr = zeros

        log_alpha = np.full(m, np.log(alpha_k + eps))
 
        feats = np.column_stack([
            cos_g, 
            log_norm_ratio, 
            margin, 
            stable_rank,
            acc,
            cos_gbar,
            log_dev_norm,
            ce_diff_nbr,
            log_alpha,
        ])

        assert feats.shape[1] == len(FEATURE_NAMES) 
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

