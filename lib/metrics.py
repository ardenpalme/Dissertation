import numpy as np
import scipy.linalg as sla
from scipy.special import softmax, log_softmax 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Compute left Perron vector and effective mixing honest submatrix
def effective_mixing(W, H, gamma_C):
    Wh = W[np.ix_(H, H)]
    h = len(H)

    #  W_bar = E[What^k]
    Wbar = (1.0 - gamma_C) * Wh
    np.fill_diagonal(Wbar, 0.0)
    np.fill_diagonal(Wbar, 1.0 - Wbar.sum(1))

    # Left Perron vector pi
    ev, evec = sla.eig(Wbar.T)
    j = int(np.argmin(np.abs(ev - 1.0)))
    if abs(ev[j] - 1.0) > 1e-8:
        raise ValueError("honest subgraph disconnected")
    pi = np.real(evec[:, j])
    pi = pi / pi.sum()
    if pi.min() <= 0:
        raise ValueError("honest subgraph reducible")

    sp = np.sqrt(pi)
    Pi = np.eye(h) - np.outer(np.ones(h), pi)
    Msim = (sp[:, None] * ((Wbar - np.outer(np.ones(h), pi)) @ Pi)) / sp[None, :]
    lam_pi = np.linalg.norm(Msim, 2)

    return Wbar, pi, lam_pi

class MetricsCalculator():
    def __init__(self, config, global_dataset, rng):
        self.X_train = global_dataset['X_train']
        self.y_train = global_dataset['y_train']
        self.X_test = global_dataset['X_test']
        self.y_test = global_dataset['y_test']
        self.C = global_dataset['num_classes']

        self.reg_param = config['reg_param']
        self.alpha_init = config['alpha_init']
        self.rng = rng

    @staticmethod
    def calc_local_opt(X_H, y_H, H, pi, dp, reg_param, rng):
        lr_args = dict(
            solver='lbfgs',
            l1_ratio=0,
            C=1/reg_param,
            fit_intercept=False, 
            max_iter=1000,
            random_state=np.random.RandomState(rng.integers(1_000_000))
        )
        
        pi_sample_weights = []
        unif_sample_weights = []
        for node_idx, pi_i in zip(H, pi):
            idx = dp[node_idx]
            pi_sample_weights.extend([pi_i / len(idx)] * len(idx))
            unif_sample_weights.extend([(1/len(H)) / len(idx)] * len(idx))
        
        clf = LogisticRegression(**lr_args)
        clf.fit(X_H, y_H, sample_weight=unif_sample_weights)
        clf_pi = LogisticRegression(**lr_args)
        clf_pi.fit(X_H, y_H, sample_weight=pi_sample_weights)

        l2_penalty = lambda x: 0.5 * reg_param * np.linalg.norm(x)**2

        return {
            'x_star': clf.coef_.T,
            'x_pi_star': clf_pi.coef_.T,
            'F_star': log_loss(y_H, clf.predict_proba(X_H)) + l2_penalty(clf.coef_),
            'F_pi_star': log_loss(y_H, clf_pi.predict_proba(X_H)) + l2_penalty(clf_pi.coef_),
        }

    @staticmethod 
    def calc_sigma_sq(theta, X, y, dp, H, C):
        sigma_sq_arr = np.empty(len(H))
        for i in range(len(H)):
            node_idx = H[i]
            Xl, yl = X[dp[node_idx]], y[dp[node_idx]]
            Yl = np.eye(C)[yl]
            R = (softmax(Xl @ theta, axis=1) - Yl) 
            sq = (Xl**2).sum(1) * (R**2).sum(1)
            gbar = Xl.T @ R / len(yl)
            sigma_sq_arr[i] = float(sq.mean() - (gbar**2).sum())
        return np.max(sigma_sq_arr)

    @staticmethod 
    def calc_zeta_sq(theta, X, y, dp, H, C):
        G = np.empty((len(H), *theta.shape))
        for i in range(len(H)):
            node_idx = H[i]
            Xl, yl = X[dp[node_idx]], y[dp[node_idx]]
            Yl = np.eye(C)[yl]
            G[i] = Xl.T @ (softmax(Xl @ theta, axis=1) - Yl) / len(yl)
        grad_hetero = np.linalg.norm(G - G.mean(0))**2
        return grad_hetero / len(H)

    @staticmethod
    def calc_zeta_sq_pi(theta, X, y, dp, H, pi, C):
        G = np.empty((len(H), *theta.shape))
        for i, node_idx in enumerate(H):
            Xl, yl = X[dp[node_idx]], y[dp[node_idx]]
            Yl = np.eye(C)[yl]
            G[i] = Xl.T @ (softmax(Xl @ theta, axis=1) - Yl) / len(yl)
        G_pi = np.tensordot(pi, G, axes=1)
        D = G - G_pi
        return float(pi @ (D ** 2).sum(axis=(1, 2)))

    @staticmethod 
    def calc_L_mu(theta_stack, X_shards, reg_param):
        Ls, mus = [], []
        for _, Xi in zip(theta_stack, X_shards):
            Ls.append(reg_param + 0.5 * np.linalg.eigvalsh(Xi.T @ Xi / Xi.shape[0]).max())
            mus.append(reg_param)
        return float(max(Ls)), float(min(mus))

    @staticmethod
    def nu2_pi_tot(W, H, pi, gamma_C):
        Wh = W[np.ix_(H, H)]
        h = len(H)
        Q = np.zeros((h, h))
        for a in range(h):
            for c in range(h):
                if a != c and Wh[a, c] > 0:
                    e = np.zeros(h)
                    e[a] = 1.0
                    e[c] = -1.0
                    Q += pi[a] * Wh[a, c] ** 2 * np.outer(e, e)

        Q *= gamma_C * (1.0 - gamma_C)
        nu2 = float(sla.eigh(Q, np.diag(pi), eigvals_only=True).max())
        w_max = float((Wh - np.diag(np.diag(Wh))).max())
        upper_bound = 4.0 * (pi.max() / pi.min()) * w_max * gamma_C * (1.0 - gamma_C)

        return nu2, upper_bound

    def __call__(self, sim_params, models, gamma_C, beta_C, proj_const):
        self.W, self.H, self.B, self.dp, self.pi = sim_params 
        self.X_shards = [self.X_train[self.dp[i]] for i in self.H]
        self.X_H = np.vstack(self.X_shards)
        self.y_H = np.concatenate([self.y_train[self.dp[i]] for i in self.H])

        _, _, lam_pi = effective_mixing(self.W, self.H, gamma_C)
        nu2, upper_bound = self.nu2_pi_tot(self.W, self.H, self.pi, gamma_C)
        L, mu = self.calc_L_mu(models[self.H], self.X_shards, self.reg_param)

        local_opt_res = self.calc_local_opt(self.X_H, self.y_H, self.H, self.pi, self.dp, self.reg_param, self.rng)
        x_opt = local_opt_res['x_star']
        zeta_sq = self.calc_zeta_sq(x_opt, self.X_train, self.y_train, self.dp, self.H, self.C)
        zeta_sq_pi = self.calc_zeta_sq_pi(x_opt, self.X_train, self.y_train, self.dp, self.H, self.pi, self.C)
        sigma_sq = self.calc_sigma_sq(x_opt, self.X_train, self.y_train, self.dp, self.H, self.C)
        
        s2 = lam_pi ** 2 + nu2
        g = 1.0 - s2
        
        dB_max = max(self.W[i, self.B].sum() for i in self.H) if len(self.B) else 0.0
        dB_pi = float(sum(self.pi[k] * self.W[i, self.B].sum() for k, i in enumerate(self.H))) if len(self.B) else 0.0

        n = min(self.dp[i].shape[0] for i in self.H)
        h = len(self.H)
        m = h
        a = self.alpha_init
        tau = proj_const * a

        # TODO: add as an assumption to paper
        # assert beta_C * dB_max <= (a*mu)/4, f"{beta_C * dB_max} > {(a*mu)/4}" 

        Z = min([len(self.dp[i]) for i in self.H])
        
        unif_dist = (1/h)*np.ones(h)
        opt_gap_ub = (3/(mu**2))*(
            (zeta_sq*h*np.linalg.norm(self.pi-unif_dist, 2)**2) +
            ((sigma_sq/Z)*((1/h)+np.max(self.pi)))
        )

        opt_gap_emp = ((local_opt_res['x_star'] - local_opt_res['x_pi_star'])**2).sum()

        
        eps_opt = (8*L*(1+(beta_C*dB_max))) / (mu*m*n)
        eps_byz = (8*m*tau*beta_C*dB_pi) / (a*mu*n)
        byz_cons_floor = ((proj_const**2)*beta_C*(dB_max**2))/(mu**2) 
        return dict(g=g, s2=s2,                          
                    pi=self.pi,                          # left Perron vector (lemma 1)
                    lam_pi=lam_pi,                       # exact calculation (lemma 5)
                    nu2=nu2,                             # exact calculation (lemma 2)
                    nu2_bound=upper_bound,               # Lemma 3 upper bound on nu^2
                    L=L,                                 # L-smoothness
                    mu=mu,                               # mu-convexity mu
                    tau=tau,                             # projection radius at k=0
                    h=h,                                 # number of honest nodes
                    n_loc=n,                             # minimum dirichlet shard size 
                    f_star=local_opt_res['F_star'],
                    f_pi_star=local_opt_res['F_pi_star'],
                    zeta_sq_pi=zeta_sq_pi,
                    sigma_sq=sigma_sq,
                    dB_max=dB_max, 
                    dB_pi=dB_pi,
                    opt_gap_ub=opt_gap_ub,               # theoretical upper bound of optimality gap \|x^*-x_pi^*\| (lemma 11)
                    opt_gap_emp=opt_gap_emp,             # empirical optimality gap \|x^*-x_pi^*\|
                    eps_opt=eps_opt,  
                    eps_byz=eps_byz, 
                    epsilon=eps_opt + eps_byz,           # on average stability upper bound
                    byz_cons_floor=byz_cons_floor,       # corollary 10 V^inf upper bound
                    alpha_max_conv=g/(6*L),              # a_k <= g/(6*L) forall k required for thm 9
                    alpha_max_stab=2/(mu+L))             # a_k <= 2/(mu+L) forall k required for thm 13
