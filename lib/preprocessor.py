import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_EPS = 1e-12


def _rank01(v):
    """Within-neighbourhood rank in [0,1]; scale-free across nodes and iterations."""
    m = len(v)
    if m < 2:
        return np.zeros(m)
    order = np.empty(m)
    order[np.argsort(v, kind='stable')] = np.arange(m)
    return order / (m - 1)


def _loo_median_mad(Xf):
    """Leave-one-out coordinate-wise median and MAD. Xf: (m, D) -> (m, D), (m, D).

    Leave-one-out matters: including x_j in its own reference statistic shrinks
    dev_j toward zero by a factor (m-1)/m and, more importantly, lets a colluding
    group set the reference it is measured against.
    """
    m = Xf.shape[0]
    loc = np.empty_like(Xf)
    scl = np.empty_like(Xf)
    for j in range(m):
        O = np.delete(Xf, j, axis=0)
        mu = np.median(O, axis=0)
        loc[j] = mu
        scl[j] = 1.4826 * np.median(np.abs(O - mu), axis=0)
    return loc, scl + _EPS


class MatrixSummaryFeatures(BaseEstimator, TransformerMixin):
    """Per-message features for the Byzantine classifier C.

    Three families:
      (a) marginal  - message vs. receiver's own state (original 7 features)
      (b) population - message vs. the empirical distribution of the m messages
                       received this round.  Targets ALIE, which is defined as a
                       displacement of the honest population statistics and is
                       therefore invisible to (a) by construction.
      (c) temporal   - message vs. the previous round's messages.  Targets the
                       one-step staleness and the missing minibatch noise in any
                       message synthesised from observed honest traffic.

    Usage per iteration k (families (b),(c) need unbroken history, so push()
    must be called on EVERY k, not only on k in sampled_iters):

        feat.set_context(k, models[i], g)
        Z = feat.transform(S)      # only when k in sampled_iters
        feat.push(S)               # every k
    """

    N_FEATURES = 10
    FEAT_NAMES = ['margin', 'entropy', 'log_norm_ratio', 'stable_rank',
                  'cos_g', 'ce_diff', 'acc',
                  'cos_dev_scale', 'neg_frac', 'centrality']
    HEAVY = ['log_norm_ratio', 'ce_diff', 'centrality']

    def __init__(self, node_id, X_local, y_local, alphas):
        self.X_local = X_local   # (n_loc, d) local (augmented) features
        self.y_local = y_local   # (n_loc,)  local labels
        self.node_id = node_id
        self.alphas = alphas
        self._prev_R = None      # (m, D) previous centred pseudo-gradients
        self._prev_gbar = None   # (D,)   previous neighbourhood pseudo-gradient

    def set_context(self, k, theta_ref, g):
        self.theta_ref = np.asarray(theta_ref)  # (d, C) current model
        self.g = g
        self.iter = k
        return self

    def fit(self, X, y=None):
        return self

    def push(self, X):
        """Store this round's statistics. Call every k, after transform()."""
        G = (self.theta_ref[None, ...] - X) / self.alphas[self.iter]
        Gf = G.reshape(X.shape[0], -1)
        gbar = np.median(Gf, axis=0)
        self._prev_R = Gf - gbar
        self._prev_gbar = gbar
        return self

    def reset_history(self):
        self._prev_R = None
        self._prev_gbar = None
        return self

    @staticmethod
    def _softmax_ce(logits, y):
        z = logits - logits.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -log_p[np.arange(len(y)), y].mean()

    @staticmethod
    def _shift_scores(theta_i, theta_j, eps=_EPS):
        A = theta_i.T @ theta_j  # (C, C)
        A /= (np.linalg.norm(theta_i, axis=0)[:, None] * np.linalg.norm(theta_j, axis=0)[None, :] + eps)
        C = A.shape[0]
        return np.array([np.mean(np.diag(np.roll(A, -s, axis=1))) for s in range(C)])  # (C,)

    def transform(self, X):
        m, d, C = X.shape
        eps = _EPS
        th_i, gi = self.theta_ref, self.g.ravel()
        Xl, yl = self.X_local, self.y_local

        G = (th_i[None, ...] - X) / self.alphas[self.iter]  # (m,d,C)
        Gf = G.reshape(m, -1)
        Gn = np.linalg.norm(Gf, axis=1)
        cos_g = (Gf @ gi) / (Gn * np.linalg.norm(gi) + eps)   # (m,)

        r = np.log(Gn / (np.linalg.norm(gi) + eps) + eps)
        sv = np.linalg.svd(G, compute_uv=False)  # (m, min(d,C))
        sr = (sv**2).sum(axis=1) / (sv[:, 0]**2 + eps)

        M = (Xl.T @ np.eye(C)[yl]) / len(yl)  # (d,C)
        sig = np.array([self._shift_scores(M, G[j]) for j in range(m)])   # (m,C)
        margin = sig.max(axis=1) - sig[:, 0]

        logits_i = Xl @ th_i
        pred_i = logits_i.argmax(axis=1)
        ce_i = self._softmax_ce(logits_i, yl)
        H, ce_d, acc = np.empty(m), np.empty(m), np.empty(m)
        for j in range(m):
            logits_j = Xl @ X[j]
            pred_j = logits_j.argmax(axis=1)
            p = np.bincount((pred_j - pred_i) % C, minlength=C) / len(pred_i) + eps
            H[j] = -(p * np.log(p)).sum()
            ce_d[j] = self._softmax_ce(logits_j, yl) - ce_i
            acc[j] = (pred_j == yl).mean()

        # ---- (b) population features -------------------------------------
        # ALIE sends mu_{-j} - z * sigma_{-j}, so dev_j is negative in every
        # coordinate and collinear with the dispersion vector.  Honest dev_j has
        # near-random coordinate signs and no relation to sigma.  Median/MAD
        # rather than mean/std because the receiver's own neighbourhood may
        # contain Byzantine nodes; valid while b_i < m_i / 2.
        Xf = X.reshape(m, -1)
        if m >= 3:
            loc, scl = _loo_median_mad(Xf)
            dev = Xf - loc
            dn = np.linalg.norm(dev, axis=1)

            neg_frac = (dev < 0).mean(axis=1)                       # ALIE ~ 1.0
            cos_dev_scale = ((dev * scl).sum(axis=1)
                             / (dn * np.linalg.norm(scl, axis=1) + eps))   # ALIE ~ -1
            zz = np.abs(dev) / scl
            z_cv = zz.std(axis=1) / (zz.mean(axis=1) + eps)          # ALIE ~ 0
            centrality = np.log(dn / (np.median(dn) + eps) + eps)    # ALIE < 0

            Dm = np.linalg.norm(Xf[:, None, :] - Xf[None, :, :], axis=-1)
            iu = np.triu_indices(m, 1)
            pd_scale = np.median(Dm[iu]) + eps
            np.fill_diagonal(Dm, np.inf)
            dup_gap = np.log(Dm.min(axis=1) / pd_scale + eps)        # colluders < 0

            gbar_loo = np.array([np.median(np.delete(Gf, j, axis=0), axis=0)
                                 for j in range(m)])
            cos_gbar = ((Gf * gbar_loo).sum(axis=1)
                        / (Gn * np.linalg.norm(gbar_loo, axis=1) + eps))
        else:
            neg_frac = np.full(m, 0.5)
            cos_dev_scale = np.zeros(m)
            z_cv = np.zeros(m)
            centrality = np.zeros(m)
            dup_gap = np.zeros(m)
            cos_gbar = np.zeros(m)

        # ---- (c) temporal features ---------------------------------------
        # Honest innovation is a fresh minibatch gradient -> nearly white.
        # ALIE's is -z * sigma^{k-1} -> smooth, autocorrelated, one step stale.
        gbar_now = np.median(Gf, axis=0)
        Rk = Gf - gbar_now
        if self._prev_R is not None and self._prev_R.shape == Gf.shape:
            Rp, gbar_p = self._prev_R, self._prev_gbar
            resid_ac = ((Rk * Rp).sum(axis=1)
                        / (np.linalg.norm(Rk, axis=1) * np.linalg.norm(Rp, axis=1) + eps))
            lag_align = (
                (Gf @ gbar_p) / (Gn * np.linalg.norm(gbar_p) + eps)
                - (Gf @ gbar_now) / (Gn * np.linalg.norm(gbar_now) + eps)
            )
        else:
            resid_ac = np.zeros(m)
            lag_align = np.zeros(m)

        return np.column_stack([
            margin, H, r, sr, np.abs(cos_g), cos_g, ce_d, acc,
            cos_dev_scale, neg_frac, z_cv, centrality, dup_gap, cos_gbar,
            resid_ac, lag_align,
            _rank01(r), _rank01(ce_d), _rank01(centrality),
        ])
