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
    def __init__(self, config, global_dataset, gf):
        self.num_nodes = config['train']['num_nodes']
        self.b = config['train']['b']
        self.K = config['train']['K']
        self.batch_sz = config['batch_sz']
        self.iter_sample_sz = self.K // 2
        self.gf = gf

        self.X = global_dataset['X_train']
        self.y = global_dataset['y_train']
        self.C = global_dataset['num_classes']

        self.reg_param = config['reg_param']
        self.beta_C = config['train']['beta_C']
        self.gamma_C = config['train']['gamma_C']
        self.training_attacks = config['train']['train_atks']
        self.testing_attacks = config['train']['test_atks']
        self.clf_model = config['train']['clf_model']

    def init_simulation(self, config, proj_const, seed):
        self.rng = rng(seed)
        self.sampled_iters = self.rng.choice(self.K, size=self.iter_sample_sz, replace=False)
        self.dp = dirichlet_partition(self.y, self.num_nodes, config['data_heterogeneity'], self.rng)

        self.G, self.W, self.B, self.H = self.gf.create_graph(config['graph_type'], config['graph_weights'], seed, **config['graph_args'])
        self.alphas = get_alphas(self.K, config)
        self.taus = proj_const * self.alphas

        # Shared Variables
        self.barrier = threading.Barrier(self.num_nodes) 
        self.models = np.zeros((self.num_nodes, self.X.shape[1], self.C))
        self.int_models = np.zeros((self.num_nodes, self.X.shape[1], self.C))
        self.tar_node = self.rng.choice(self.H)

    def init_preproc(self):
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        heavy = [idx[n] for n in HEAVY_FEATURES]
        bounded = [idx[n] for n in FEATURE_NAMES if idx[n] not in heavy]
 
        self.feat_pre_proc = ColumnTransformer([
            ('robust', RobustScaler(unit_variance=True), heavy),
            ('std',    StandardScaler(), bounded),
        ])
        self.out_feat_names = ([FEATURE_NAMES[i] for i in heavy]
                               + [FEATURE_NAMES[i] for i in bounded])


    def worker_RDSGD(self, i, barrier, models, int_models, alphas, taus,
                         results, beta_C, gamma_C, rng):

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
            barrier.wait() # Byzantine x^{k+1/2} written

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

    def simulate(self, atk_type, seed, beta_C, gamma_C):
        self.models.fill(0)
        self.int_models.fill(0)
        results = [None] * self.num_nodes
        
        hon_threads = [threading.Thread(target=self.worker_RDSGD, 
                        args=(i, self.barrier, self.models, self.int_models, self.alphas, self.taus, 
                              results, beta_C, gamma_C, rng('classifier', 'hon', seed, i)))
                       for i in self.H]

        byz_threads = [threading.Thread(target=byz_atk, 
                        args=(i, self.barrier, self.models, self.int_models, self.X, self.y, self.dp, 
                              self.C, self.K, self.W, self.G, self.H,
                              self.reg_param, self.batch_sz, self.alphas, atk_type, rng('classifier', 'byz', seed, i)),
                                        # TODO make ALIE, IPM args a global config param
                                        kwargs={'alie_z': None, 'num_nodes':self.num_nodes, 'b': self.b, 'ipm_eps':0.5}) 
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

    def _gather(self, attacks, seed, beta_C, gamma_C):
        Xs, ys, gs, atks = [], [], [], []

        for a_idx, atk in enumerate(attacks):
            X_C, y_C, g_C = self.simulate(atk, seed, beta_C, gamma_C)
            Xs.append(X_C)
            ys.append(y_C)
            gs.append(g_C + 1000 * a_idx) # keep attacks in distinct groups
            atks.append(np.full(len(y_C), a_idx))
        return (np.concatenate(Xs), np.concatenate(ys),
                np.concatenate(gs), np.concatenate(atks))

    def run_simulations(self, config, proj_const, sim_seed, 
                        training_attacks=None, testing_attacks=None,
                        beta=None, gamma=None, is_printing=True):
        train_sim_seed = sim_seed * 17 + 23
        val_sim_seed = sim_seed * 11 + 1
        thr_sim_seed = sim_seed * 617 + 37
        test_sim_seed = sim_seed * 217 + 31

        beta_C = beta if beta is not None else self.beta_C
        gamma_C = gamma if gamma is not None else self.gamma_C

        train_atks = training_attacks if training_attacks is not None else self.training_attacks 
        test_atks = testing_attacks if testing_attacks is not None else self.testing_attacks

        if is_printing:
            print(f"Simulating RDSGD @ (beta_C={beta_C}, gamma_C={gamma_C})")
            print(f'Building training set. Simulating {train_atks} (seed {train_sim_seed})')
        self.init_simulation(config, proj_const, train_sim_seed)
        Xtr, ytr, gtr, atr = self._gather(train_atks, train_sim_seed, beta_C, gamma_C)
 
        if is_printing: print(f'Building validation set I. Simulating {train_atks} (seed {val_sim_seed})')
        self.init_simulation(config, proj_const, val_sim_seed)
        Xva, yva, gva, ava = self._gather(train_atks, val_sim_seed, beta_C, gamma_C)

        if is_printing: print(f'Building validation set II. Simulating {train_atks} (seed {thr_sim_seed})')
        self.init_simulation(config, proj_const, thr_sim_seed)
        Xth, yth, gth, ath = self._gather(train_atks, thr_sim_seed, beta_C, gamma_C)
 
        if is_printing: print(f'Building test set. Simulating {test_atks} (seed {test_sim_seed})')
        self.init_simulation(config, proj_const, test_sim_seed)
        Xte, yte, gte, ate = self._gather(test_atks, test_sim_seed, beta_C, gamma_C)
        
        self.init_preproc()
        self.data = dict(
            X_train=self.feat_pre_proc.fit_transform(Xtr), y_train=ytr,
            groups_train=gtr, atk_train=atr,
            X_val=self.feat_pre_proc.transform(Xva), y_val=yva,
            groups_val=gva, atk_val=ava,
            X_thr=self.feat_pre_proc.transform(Xth), y_thr=yth,
            groups_thr=gth, atk_thr=ath,
            X_test=self.feat_pre_proc.transform(Xte), y_test=yte,
            groups_test=gte, atk_test=ate,
            attack_names=list(train_atks),
            test_attack_names=list(test_atks))
        return self.data

    def fit(self, in_data=None, in_model=None):
        data = in_data if in_data is not None else self.data
        X_train, y_train, g_train = data['X_train'], data['y_train'], data['groups_train']
        model = in_model if in_model is not None else self.clf_model

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
 
    def calc_optimal_op_pt(self, in_data=None, cost_fp=1.0, cost_fn=3.0):
        data = in_data if in_data is not None else self.data
        y = data['y_thr']
        p = self.best_est.predict_proba(data['X_thr'])[:, 1]
        fpr, tpr, thr = roc_curve(y, p)
        auc = roc_auc_score(y, p)

        pi = float(y.mean())
        cost = pi * cost_fn * (1 - tpr) + (1 - pi) * cost_fp * fpr
        idx = int(np.argmin(cost))

        self.opt_fpr = float(fpr[idx])
        self.opt_tpr = float(tpr[idx])
        self.opt_fnr = float(1 - tpr[idx])
        self.opt_tau = float(np.clip(thr[idx], 0.0, 1.0))
 
        return {'C_fpr': self.opt_fpr, 
                'C_fnr': self.opt_fnr,
                'C_tau': self.opt_tau, 
                'C_auc': auc,
                'cv_ap': self.cv_score,
                'cv_params': self.cv_params}

    def test(self, in_data=None):
        d= in_data if in_data is not None else self.data
        X_C, y_C = d['X_test'], d['y_test'] 
 
        p = self.best_est.predict_proba(X_C)[:, 1]
        self.fpr, self.tpr, self.roc_thr = roc_curve(y_C, p)
        self.auc = roc_auc_score(y_C, p)
        self.prec, self.rec, self.pr_thr = precision_recall_curve(y_C, p)
        self.ap = average_precision_score(y_C, p)
        self.prevalence = float(y_C.mean())
        j = int(np.clip(np.searchsorted(self.pr_thr, self.opt_tau), 0, len(self.prec) - 1))
        self.op_rec = self.rec[j]
        self.op_prec = self.prec[j]
 
        tn, fp, fn, tp = confusion_matrix(y_C, (p >= self.opt_tau).astype(int)).ravel()
        return {'prevalence': self.prevalence, 'tau': self.opt_tau,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                'op_prec':self.op_prec, 'op_rec': self.op_rec,
                'beta_C': fn / max(fn + tp, 1), 'gamma_C': fp / max(fp + tn, 1),
                'roc_auc': self.auc, 'avg_prec': self.ap}

    def plot_roc(self, ax):
        # ROC curve from validation dataset
        ax.plot(self.fpr, self.tpr, label=f'ROC curve (AUC = {self.auc:.3f})', linewidth=2)
        ax.scatter([self.opt_fpr], [self.opt_tpr], color='red', zorder=5, 
                   label=f'Operating point (τ = {self.opt_tau:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Byzantine Classifier ROC Curve (val)')
        ax.legend(loc='lower right')

    def plot_pr(self,ax):
        # PR curve from test dataset
        ax.plot(self.rec, self.prec, label=f'PR curve (AP = {self.ap:.3f})', linewidth=2)
        ax.axhline(self.prevalence, color='k', linestyle='--', 
                   label=f'Random classifier ({self.prevalence:.3f})')
        ax.scatter([self.op_rec], [self.op_prec], color='red', zorder=5, 
                   label=f'Operating point (τ = {self.opt_tau:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (1 − FNR)')
        ax.set_ylabel('Precision')
        ax.set_title('Byzantine Classifier PR Curve (test)')
        ax.legend(loc='lower left')
