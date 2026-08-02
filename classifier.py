import threading
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, log_loss
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from graph_factory import GraphFactory
from utils import rng, seed, sgd_grad, proj_tau, dirichlet_partition, get_alphas
from dist_alg import byz_atk

class MatrixSummaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, node_id, X_local, y_local, alphas):
        self.X_local = X_local   # (n_loc, d) local (augmented) features
        self.y_local = y_local   # (n_loc,)  local labels
        self.node_id = node_id
        self.alphas = alphas

    def set_context(self, k, theta_ref, g):
        self.theta_ref = np.asarray(theta_ref)  # (d, C) current model
        self.g = g
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
        m, d, C = X.shape
        eps = 1e-12
        th_i, gi = self.theta_ref, self.g.ravel()
        Xl, yl = self.X_local, self.y_local

        G = (th_i[None, ...] - X) / self.alphas[self.iter]  # (m,d,C)
        Gf = G.reshape(m, -1)
        Gn = np.linalg.norm(Gf, axis=1)
        cos_g = (Gf @ gi) / (np.linalg.norm(Gf, axis=1) * np.linalg.norm(gi) + eps)   # (m,)
        
        r = np.log(Gn / (np.linalg.norm(gi) + eps) + eps)
        sv = np.linalg.svd(G, compute_uv=False) # (m, min(d,C))
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
            pred_j   = logits_j.argmax(axis=1)
            p = np.bincount((pred_j - pred_i) % C, minlength=C) / len(pred_i) + eps
            H[j] = -(p * np.log(p)).sum()
            ce_d[j] = self._softmax_ce(logits_j, yl) - ce_i
            acc[j] = (pred_j == yl).mean()

        return np.column_stack([margin, H, r, sr, np.abs(cos_g), ce_d, acc])

class ByzClassifier():
    def __init__(self, config, global_dataset, rng):
        self.num_nodes = config['train']['num_nodes']
        self.b = config['train']['b']
        self.K = config['train']['K']
        self.batch_sz = config['batch_sz']
        self.iter_sample_sz = self.K//2
        self.rng = rng

        self.X = global_dataset['X_train']
        self.y = global_dataset['y_train']
        self.C = global_dataset['num_classes']

        self.reg_param = config['reg_param']
        self.sim_beta_C = config['train']['beta_C']
        self.sim_gamma_C = config['train']['gamma_C']
        self.training_attacks = config['train']['train_atks']
        self.testing_attacks = config['train']['val_atks']

    def configure(self, gf, config, proj_const):
        self.sampled_iters = self.rng.choice(self.K, size=self.iter_sample_sz, replace=False)
        self.dp = dirichlet_partition(self.y, self.num_nodes, config['data_heterogeneity'], self.rng)
        self.G, self.W, sampling_subset = gf.create_graph(config['graph_type'], config['graph_weights'], **config['graph_args'])
        self.B = self.rng.choice(sampling_subset, size=self.b, replace=False)
        self.H = np.array(list(set(np.arange(self.num_nodes)) - set(self.B)))
        self.alphas = get_alphas(self.K, config)
        self.taus = proj_const * self.alphas

        # Shared Variables
        self.barrier = threading.Barrier(self.num_nodes) 
        self.models = np.zeros((self.num_nodes, self.X.shape[1], self.C))
        self.int_models = np.zeros((self.num_nodes, self.X.shape[1], self.C))

    def worker_DSGD(self, i, barrier, models, int_models, alphas, taus,
                         results, rng, beta_C=0.1, gamma_C=0.1):

        Xl, yl = self.X[self.dp[i]], self.y[self.dp[i]]
        feat = MatrixSummaryFeatures(i, Xl, yl, alphas)
        nbors = list(self.G.neighbors(i))
        byz_nbors = np.isin(nbors, self.B)
        hon_nbors = np.isin(nbors, self.H)
        feats, labels = [], []
        for k in range(self.K):
            _, g = sgd_grad(Xl, yl, models[i], self.reg_param, self.batch_sz, rng)
            int_models[i] = models[i] - alphas[k] * g
            barrier.wait()

            if k in self.sampled_iters:
                feat.set_context(k, models[i], g)
                Z = feat.transform(np.stack([int_models[j] for j in nbors]))
                feats.append(Z)
                labels.extend(int(j in self.B) for j in nbors)
                
            drop = np.where((byz_nbors & (rng.random(len(nbors)) < (1-beta_C))) | (hon_nbors & (rng.random(len(nbors)) < gamma_C)))
            w_row = self.W[i].copy()
            w_row[np.asarray(nbors)[drop]] = 0.0
            w_row[i] = 1.0 - w_row[nbors].sum()
            models[i] = sum(w_row[j] * proj_tau(models[i], int_models[j], taus[k]) for j in nbors + [i]) 
            barrier.wait()
        results[i] = (np.concatenate(feats), np.array(labels))

    def simulate(self, atk_type, sim_id):
        self.models.fill(0)
        self.int_models.fill(0)
        results = [None] * self.num_nodes
        
        hon_threads = [threading.Thread(target=self.worker_DSGD, 
                        args=(i, self.barrier, self.models, self.int_models, self.alphas, self.taus, 
                              results, rng('classifier', 'hon', sim_id, i)),
                        kwargs={'beta_C': self.sim_beta_C, 
                                'gamma_C':self.sim_gamma_C})
                       for i in self.H]
        
        byz_threads = [threading.Thread(target=byz_atk, 
                        args=(i, self.barrier, self.models, self.int_models, 
                              self.X[self.dp[i]], self.y[self.dp[i]], self.C, self.K, self.W, self.G, 
                              self.reg_param, self.batch_sz, self.alphas, atk_type, rng('classifier', 'byz', sim_id, i)))
                       for i in self.B]
        
        threads = hon_threads + byz_threads
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        X_C = np.concatenate([results[i][0] for i in self.H])
        y_C = np.concatenate([results[i][1] for i in self.H])
        return X_C, y_C

    def run_simulations(self):
        # Pre-process run iterates
        feat_names = ['margin', 'entropy', 'log_norm_ratio', 'stable_rank', 'abs_cos_g', 'ce_diff', 'acc']
        feature_idx = {n: i for i, n in enumerate(feat_names)}

        heavy = [feature_idx['log_norm_ratio'], feature_idx['ce_diff']]
        bounded = [feature_idx[n] for n in feat_names if feature_idx[n] not in heavy]

        self.feat_pre_proc = ColumnTransformer([
            ('robust', RobustScaler(unit_variance=True), heavy),
            ('std',    StandardScaler(), bounded),
        ])
        self.out_feat_names = ['log_norm_ratio', 'ce_diff', 'margin', 'entropy', 'stable_rank', 'abs_cos_g', 'acc']

        # Gather data through simulated DSGD runs
        train_covariates, train_labels = [], []
        test_covariates, test_labels = [], []

        for atk in self.training_attacks:
            X_C, y_C = self.simulate(atk, 1)
            train_covariates.append(X_C)
            train_labels.append(y_C)

        for atk in self.training_attacks:
            X_C, y_C = self.simulate(atk, 2)
            test_covariates.append(X_C)
            test_labels.append(y_C)
            
        X_C_train = self.feat_pre_proc.fit_transform(np.concatenate(train_covariates))
        y_C_train = np.concatenate(train_labels)

        X_C_test = self.feat_pre_proc.transform(np.concatenate(test_covariates))
        y_C_test = np.concatenate(test_labels)

        return X_C_train, y_C_train, X_C_test, y_C_test

    def train_and_eval(self, X_C_train, y_C_train, X_C_test, y_C_test):
        lr_clf = LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs', 
                            random_state=seed(2, 'log-reg'))
        grid = GridSearchCV(lr_clf, param_grid={'C': [0.01, 0.1, 1.0, 10.0]}, cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_C_train, y_C_train)

        self.best_est = grid.best_estimator_
        y_C_pred = self.best_est.predict(X_C_test)
        y_C_proba = self.best_est.predict_proba(X_C_test)[:, 1]

        self.auc = roc_auc_score(y_C_test, y_C_proba)
        self.fpr, self.tpr, self.thr = roc_curve(y_C_test, y_C_proba)

        return {
            'test_acc': accuracy_score(y_C_test, y_C_pred),
            'recall': recall_score(y_C_test, y_C_pred),
            'roc-auc': self.auc,
            'f1-score': f1_score(y_C_test, y_C_pred)
        }

    def get_params(self):
        return self.best_est, self.feat_pre_proc

    def clac_opt_operating_pt(self, quantile):
        idx = np.searchsorted(self.tpr, quantile, side='left')
        self.opt_fpr = self.fpr[idx]
        self.opt_fnr = 1-self.tpr[idx]
        self.opt_tau = self.thr[idx]
        specs = {
            'fpr':self.opt_fpr,
            'fnr':self.opt_fnr,
            'tau':self.opt_tau,
        }
        return specs

    def plot(self, ax):
        ax.plot(self.fpr, self.tpr, label=f'ROC curve (AUC = {self.auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Byzantine Classifier ROC Curve')
        ax.legend(loc='lower right')



