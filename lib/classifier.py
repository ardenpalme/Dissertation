import threading
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.kernel_approximation import Nystroem
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, train_test_split, cross_validate, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, log_loss, precision_recall_curve, average_precision_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from graph_factory import GraphFactory
from utils import rng, seed, sgd_grad, proj_tau, dirichlet_partition, get_alphas
from dist_alg import byz_atk
from preprocessor import MatrixSummaryFeatures, FEATURE_NAMES, HEAVY_FEATURES
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

class ByzClassifier():
    def __init__(self, config, global_dataset, gf, rng):
        self.num_nodes = config['train']['num_nodes']
        self.b = config['train']['b']
        self.K = config['train']['K']
        self.batch_sz = config['batch_sz']
        self.iter_sample_sz = self.K // 2
        self.rng = rng
        self.gf = gf

        self.X = global_dataset['X_train']
        self.y = global_dataset['y_train']
        self.C = global_dataset['num_classes']

        self.reg_param = config['reg_param']
        self.sim_beta_C = config['train']['beta_C']
        self.sim_gamma_C = config['train']['gamma_C']
        self.training_attacks = config['train']['train_atks']
        self.testing_attacks = config['train']['val_atks']

    def init_simulation(self, config, proj_const):
        self.sampled_iters = self.rng.choice(self.K, size=self.iter_sample_sz, replace=False)
        self.dp = dirichlet_partition(self.y, self.num_nodes, config['data_heterogeneity'], self.rng)
        self.G, self.W, sampling_subset = self.gf.create_graph(config['graph_type'], config['graph_weights'], **config['graph_args'])
        self.B = self.rng.choice(sampling_subset, size=self.b, replace=False)
        self.H = np.array(list(set(np.arange(self.num_nodes)) - set(self.B)))
        self.alphas = get_alphas(self.K, config)
        self.taus = proj_const * self.alphas

        # Shared Variables
        self.barrier = threading.Barrier(self.num_nodes) 
        self.models = np.zeros((self.num_nodes, self.X.shape[1], self.C))
        self.int_models = np.zeros((self.num_nodes, self.X.shape[1], self.C))
        self.tar_node = self.rng.choice(self.H)

    def init_preproc(self):
        """Column transformer over the canonical feature ordering.
 
        Names and heavy/bounded membership are derived from preprocessor.py so
        they cannot drift out of sync with transform().  Note ColumnTransformer
        emits the robust block first, so out_feat_names is reordered to match.
        """
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        heavy = [idx[n] for n in HEAVY_FEATURES]
        bounded = [idx[n] for n in FEATURE_NAMES if idx[n] not in heavy]
 
        self.feat_pre_proc = ColumnTransformer([
            ('robust', RobustScaler(unit_variance=True), heavy),
            ('std',    StandardScaler(), bounded),
        ])
        self.out_feat_names = ([FEATURE_NAMES[i] for i in heavy]
                               + [FEATURE_NAMES[i] for i in bounded])


    def worker_RDSGD(self, i, tar_node, barrier, models, int_models, alphas, taus,
                         results, rng, beta_C=0.1, gamma_C=0.1):

        Xl, yl = self.X[self.dp[i]], self.y[self.dp[i]]
        feat = MatrixSummaryFeatures(i, Xl, yl, alphas)
        nbors = list(self.G.neighbors(i))
        byz_nbors = np.isin(nbors, self.B)
        hon_nbors = np.isin(nbors, self.H)

        feats, labels, groups = [], [], []
        for k in range(self.K):
            _, g = sgd_grad(Xl, yl, models[i], self.reg_param, self.batch_sz, rng)
            int_models[i] = models[i] - alphas[k] * g
            barrier.wait()

            if k in self.sampled_iters:
                feat.set_context(k, models[i], g)
                Z = feat.transform(np.stack([int_models[j] for j in nbors]))
                feats.append(Z)
                labels.extend(int(j in self.B) for j in nbors)
                groups.extend([i] * len(nbors))       # group = originating node
                
            drop = np.where((byz_nbors & (rng.random(len(nbors)) < (1-beta_C))) 
                            | (hon_nbors & (rng.random(len(nbors)) < gamma_C)))
            w_row = self.W[i].copy()
            w_row[np.asarray(nbors)[drop]] = 0.0
            w_row[i] = 1.0 - w_row[nbors].sum()
            models[i] = sum(w_row[j] * proj_tau(models[i], int_models[j], taus[k]) for j in nbors + [i]) 
            barrier.wait()
            barrier.wait()

        results[i] = (np.concatenate(feats), np.array(labels), np.array(groups))

    def simulate(self, atk_type, sim_id, beta_C, gamma_C):
        self.models.fill(0)
        self.int_models.fill(0)
        results = [None] * self.num_nodes
        
        hon_threads = [threading.Thread(target=self.worker_RDSGD, 
                        args=(i, self.tar_node, self.barrier, self.models, self.int_models, self.alphas, self.taus, 
                              results, rng('classifier', 'hon', sim_id, i)),
                        kwargs={'beta_C' : beta_C, 'gamma_C' : gamma_C})
                       for i in self.H]

        byz_threads = [threading.Thread(target=byz_atk, 
                        args=(i, self.barrier, self.models, self.int_models, self.X, self.y, self.dp, 
                              self.C, self.K, self.W, self.G, self.H,
                              self.reg_param, self.batch_sz, self.alphas, atk_type, rng('classifier', 'byz', sim_id, i)),
                                        kwargs={'alie_z': None, 'num_nodes':self.num_nodes, 'b': self.b,
                                                'ipm_eps':0.5})
                       for i in self.B]
        
        threads = hon_threads + byz_threads
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        X_C = np.concatenate([results[i][0] for i in self.H])
        y_C = np.concatenate([results[i][1] for i in self.H])
        g_C = np.concatenate([results[i][2] for i in self.H])
        return X_C, y_C, g_C

    def _gather(self, attacks, sim_id, beta_C=None, gamma_C=None):
        Xs, ys, gs, atks = [], [], [], []
        sim_beta_C = beta_C if beta_C is not None else self.sim_beta_C
        sim_gamma_C = gamma_C if beta_C is not None else self.sim_gamma_C

        for a_idx, atk in enumerate(attacks):
            X_C, y_C, g_C = self.simulate(atk, sim_id, sim_beta_C, sim_gamma_C)
            Xs.append(X_C)
            ys.append(y_C)
            gs.append(g_C + 1000 * a_idx)          # keep attacks in distinct groups
            atks.append(np.full(len(y_C), a_idx))
        return (np.concatenate(Xs), np.concatenate(ys),
                np.concatenate(gs), np.concatenate(atks))

    def run_simulations(self, config, proj_const, sim_seed, beta_C=0.1, gamma_C=0.05, is_printing=True):
        train_sim_seed = sim_seed * 17 + 23
        val_sim_seed = sim_seed * 11 + 1
        test_sim_seed = sim_seed * 217 + 31

        if is_printing:
            print(f"Simulating {self.training_attacks}(beta={beta_C}, gamma={gamma_C})")
            print(f'  train set (sim {train_sim_seed})')
        self.init_simulation(config, proj_const)
        Xtr, ytr, gtr, atr = self._gather(self.training_attacks, train_sim_seed, beta_C, gamma_C)
 
        if is_printing: print(f'  validation set (sim {val_sim_seed})')
        self.init_simulation(config, proj_const)
        Xva, yva, gva, ava = self._gather(self.training_attacks, val_sim_seed, beta_C, gamma_C)
 
        if is_printing: print(f'  test set (sim {test_sim_seed})')
        self.init_simulation(config, proj_const)
        Xte, yte, gte, ate = self._gather(self.testing_attacks, test_sim_seed, beta_C, gamma_C)
        
        self.init_preproc()
        self.data = dict(
            X_train=self.feat_pre_proc.fit_transform(Xtr), y_train=ytr,
            groups_train=gtr, atk_train=atr,
            X_val=self.feat_pre_proc.transform(Xva), y_val=yva,
            groups_val=gva, atk_val=ava,
            X_test=self.feat_pre_proc.transform(Xte), y_test=yte,
            groups_test=gte, atk_test=ate,
            attack_names=list(self.training_attacks),
            test_attack_names=list(self.testing_attacks))
        return self.data

    def train_and_eval(self, in_data=None, model='lin-lr'):
        data = in_data if in_data is not None else self.data
        X_train, y_train, g_train = data['X_train'], data['y_train'], data['groups_train']

        # Group-aware folds: rows from one node share that node's local shard,
        # which ce_diff / acc / margin are computed against, so shuffled folds
        # leak and inflate the CV score.
        n_groups = len(np.unique(g_train))
        cv = GroupKFold(n_splits=min(5, n_groups))
        splits = list(cv.split(X_train, y_train, g_train))

        MODELS = {
            'lin-lr': (
                LogisticRegression(class_weight='balanced', max_iter=1000,
                                   solver='lbfgs', random_state=seed(2, 'log-reg')),
                {'C': [0.01, 0.1, 1.0, 10.0]},
            ),
            'rbf': (
                Pipeline([
                    ('rbf', Nystroem(random_state=seed('classifier', 'nystroem'))),
                    ('lr',  LogisticRegression(class_weight='balanced', max_iter=2000,
                                               random_state=seed(2, 'log-reg'))),
                ]),
                {'rbf__gamma': [0.02, 0.05, 0.1, 0.3],
                 'rbf__n_components': [150, 400],
                 'lr__C': [0.1, 1.0, 10.0]},
            ),
            'mlp': (
                MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                              max_iter=400, random_state=seed('classifier', 'mlp')),
                {'alpha': [1e-4, 1e-2], 'learning_rate_init': [1e-3, 3e-3]},
            ),
            'rf':(
                RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample',
                           min_samples_leaf=50, max_features='sqrt',
                           n_jobs=1, random_state=seed('classifier', 'rf')),
                {'max_depth': [6, 12, None], 'min_samples_leaf': [20, 50, 200],
                 'max_features': ['sqrt', 0.5]},
            ),
            'xgb': (
                XGBClassifier(tree_method='hist',
                              scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
                              eval_metric='aucpr', n_estimators=500,
                              random_state=seed('classifier', 'xgb')),
                {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1],
                 'min_child_weight': [50, 200], 'reg_lambda': [1.0, 10.0]},
            ),
        }
        est, param_grid = MODELS[model]
 
        grid = GridSearchCV(est, param_grid=param_grid, cv=splits, scoring='average_precision', n_jobs=-1)
        grid.fit(X_train, y_train)
        self.best_est = grid.best_estimator_
        self.cv_score = grid.best_score_
        self.cv_params = grid.best_params_

        return self.best_est, self.feat_pre_proc
 
    def calc_optimal_op_pt(self, in_data=None, quantile=0.8):
        data = in_data if in_data is not None else self.data

        # Operating point from the held-out validation simulation.
        p_val = self.best_est.predict_proba(data['X_val'])[:, 1]
        fpr, tpr, thr = roc_curve(data['y_val'], p_val)
        idx = int(np.clip(np.searchsorted(tpr, quantile, side='left'), 0, len(thr) - 1))
        self.opt_fpr = float(fpr[idx])
        self.opt_fnr = float(1 - tpr[idx])
        self.opt_tau = float(thr[idx])
 
        return {'C_fpr': self.opt_fpr, 'C_fnr': self.opt_fnr,
                'C_tau': self.opt_tau, 'cv_ap': self.cv_score,
                'cv_params': self.cv_params}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def test(self, X_C=None, y_C=None):
        d = self.data
        X_C = d['X_test'] if X_C is None else X_C
        y_C = d['y_test'] if y_C is None else y_C
 
        p = self.best_est.predict_proba(X_C)[:, 1]
        self.fpr, self.tpr, self.roc_thr = roc_curve(y_C, p)
        self.auc = roc_auc_score(y_C, p)
        self.prec, self.rec, self.pr_thr = precision_recall_curve(y_C, p)
        self.ap = average_precision_score(y_C, p)
        self.prevalence = float(y_C.mean())
 
        j = int(np.clip(np.searchsorted(self.pr_thr, self.opt_tau), 0, len(self.prec) - 1))
        self.op_prec, self.op_rec = self.prec[j], self.rec[j]
 
        tn, fp, fn, tp = confusion_matrix(y_C, (p >= self.opt_tau).astype(int)).ravel()
        return {'prevalence': self.prevalence, 'tau': self.opt_tau,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                'beta_C': fn / max(fn + tp, 1), 'gamma_C': fp / max(fp + tn, 1),
                'roc_auc': self.auc, 'avg_prec': self.ap}

    def per_attack_report(self, split='test'):
        """Recall and FPR per attack at the deployed threshold.
 
        Pooled AUC has repeatedly looked healthy while one attack's recall was
        poor; this is the table that predicts RDSGD's downstream accuracy.
        """
        d = self.data
        X, y, atk = d[f'X_{split}'], d[f'y_{split}'], d[f'atk_{split}']
        names = (d['test_attack_names'] if split == 'test' else d['attack_names'])
        p = self.best_est.predict_proba(X)[:, 1]
        pred = p >= self.opt_tau
 
        rows = []
        for a in np.unique(atk):
            m = atk == a
            neg = m & (y == 0)
            rows.append({
                'attack': names[a] if a < len(names) else str(a),
                'n': int(m.sum()),
                'prevalence': float(y[m].mean()),
                'recall': float(recall_score(y[m], pred[m], zero_division=0)),
                'beta_C': float(1 - recall_score(y[m], pred[m], zero_division=0)),
                'gamma_C': float(pred[neg].mean()) if neg.any() else np.nan,
                'roc_auc': (float(roc_auc_score(y[m], p[m]))
                            if len(np.unique(y[m])) > 1 else np.nan)})
        df = pd.DataFrame(rows).set_index('attack')
        self.worst_recall = float(df['recall'].min())
        return df
 
    def recall_by_iteration(self, split='test', k_col='k_frac', bins=7):
        """Recall as a function of training progress.
 
        Requires k_frac to survive preprocessing (it does -- it is in the
        bounded block).  A declining curve is the signature of the alpha_k
        contamination described in preprocessor.MatrixSummaryFeatures.
        """
        d = self.data
        X, y = d[f'X_{split}'], d[f'y_{split}']
        col = self.out_feat_names.index(k_col)
        p = self.best_est.predict_proba(X)[:, 1]
        pred = p >= self.opt_tau
 
        q = pd.qcut(X[:, col], bins, labels=False, duplicates='drop')
        rows = []
        for b in np.unique(q):
            m = (q == b) & (y == 1)
            n = (q == b) & (y == 0)
            rows.append({'bin': int(b),
                         'recall': float(pred[m].mean()) if m.any() else np.nan,
                         'fpr': float(pred[n].mean()) if n.any() else np.nan})
        return pd.DataFrame(rows).set_index('bin')

    def plot_roc(self, ax):
        ax.plot(self.fpr, self.tpr, label=f'ROC curve (AUC = {self.auc:.3f})', linewidth=2)
        ax.scatter([self.opt_fpr], [1 - self.opt_fnr], color='red', zorder=5, 
                   label=f'Operating point (τ = {self.opt_tau:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Byzantine Classifier ROC Curve')
        ax.legend(loc='lower right')

    def plot_pr(self,ax):
        ax.plot(self.rec, self.prec, label=f'PR curve (AP = {self.ap:.3f})', linewidth=2)
        ax.axhline(self.prevalence, color='k', linestyle='--', 
                   label=f'Random classifier ({self.prevalence:.3f})')
        ax.scatter([self.op_rec], [self.op_prec], color='red', zorder=5, 
                   label=f'Operating point (τ = {self.opt_tau:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (1 − FNR)')
        ax.set_ylabel('Precision')
        ax.set_title('Byzantine Classifier PR Curve')
        ax.legend(loc='lower left')


    def plot_feature_importance(self, X_te, y_te, ax):
        names = list(self.out_feat_names)
        assert len(names) == X_te.shape[1], f"{len(names)} names vs {X_te.shape[1]} columns"

        r = permutation_importance(self.best_est, X_te, y_te, n_repeats=30, scoring='roc_auc', n_jobs=-1)

        long = (pd.DataFrame(r.importances.T, columns=names).melt(var_name='feature', value_name='importance'))
        order = r.importances_mean.argsort()[::-1]
        order = [names[i] for i in order]

        sns.violinplot(data=long, x='importance', y='feature', order=order, orient='h', color='steelblue', ax=ax)
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.set(xlabel=f'Drop in roc-auc when permuted', title='Feature importance')
