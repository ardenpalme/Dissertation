import threading
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, log_loss, precision_recall_curve, average_precision_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from graph_factory import GraphFactory
from utils import rng, seed, sgd_grad, proj_tau, dirichlet_partition, get_alphas
from dist_alg import byz_atk
from preprocessor import MatrixSummaryFeatures as MSF

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

    def init_preproc(self):
        feat_names  = MSF.FEAT_NAMES
        feature_idx = {n: i for i, n in enumerate(feat_names)}
        assert len(feat_names) == MSF.N_FEATURES

        heavy    = [n for n in feat_names if n in MSF.HEAVY]
        bounded  = [n for n in feat_names if n not in heavy ]

        self.feat_pre_proc = ColumnTransformer([
            ('robust', RobustScaler(unit_variance=True), [feature_idx[n] for n in heavy]),
            ('std',    StandardScaler(),                 [feature_idx[n] for n in bounded]),
        ])
        self.out_feat_names = heavy + bounded 

    def worker_DSGD(self, i, barrier, models, int_models, alphas, taus,
                         results, rng, beta_C=0.1, gamma_C=0.1):

        Xl, yl = self.X[self.dp[i]], self.y[self.dp[i]]
        feat = MSF(i, Xl, yl, alphas)
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
                        args=(i, self.barrier, self.models, self.int_models, self.X, self.y, self.dp, 
                              self.C, self.K, self.W, self.G, self.H,
                              self.reg_param, self.batch_sz, self.alphas, self.taus, atk_type, rng('classifier', 'byz', sim_id, i)),
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
        return X_C, y_C

    def run_simulations(self):
        # Gather data through simulated DSGD runs
        train_covariates, train_labels = [], []
        test_covariates, test_labels = [], []

        for atk in self.training_attacks:
            X_C, y_C = self.simulate(atk, 1)
            train_covariates.append(X_C)
            train_labels.append(y_C)

        for atk in self.training_attacks:
            X_C, y_C = self.simulate(atk, 29)
            test_covariates.append(X_C)
            test_labels.append(y_C)

        self.init_preproc()
            
        X_C_train = self.feat_pre_proc.fit_transform(np.concatenate(train_covariates))
        y_C_train = np.concatenate(train_labels)

        X_C_test = self.feat_pre_proc.transform(np.concatenate(test_covariates))
        y_C_test = np.concatenate(test_labels)

        return X_C_train, y_C_train, X_C_test, y_C_test

    def train_and_eval(self, X_C, y_C, quantile=0.8):
        X_C_train, X_C_val, y_C_train, y_C_val = train_test_split(X_C, y_C, test_size=0.33, random_state=seed('classifier','tr-val-split'))

        lr_clf = LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs', 
                            random_state=seed(2, 'log-reg'))
        grid = GridSearchCV(lr_clf, param_grid={'C': [0.01, 0.1, 1.0, 10.0]}, cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_C_train, y_C_train)
        self.best_est = grid.best_estimator_

        # return the best operating point based on validation dataset
        y_C_proba = self.best_est.predict_proba(X_C_val)[:, 1]
        fpr, tpr, roc_thr = roc_curve(y_C_val, y_C_proba)
        idx = np.searchsorted(tpr, quantile, side='left')
        self.opt_fpr = fpr[idx]
        self.opt_fnr = 1-tpr[idx]
        self.opt_tau = roc_thr[idx]

        return {
            'C_fpr':self.opt_fpr,
            'C_fnr':self.opt_fnr,
            'C_tau':self.opt_tau,
        }


    def get_params(self):
        return self.best_est, self.feat_pre_proc

    def test(self, X_C, y_C):
        y_C_proba = self.best_est.predict_proba(X_C)[:, 1]
        self.fpr, self.tpr, self.roc_thr = roc_curve(y_C, y_C_proba)
        self.auc = roc_auc_score(y_C, y_C_proba)
        self.prec, self.rec, self.pr_thr = precision_recall_curve(y_C, y_C_proba)
        self.ap = average_precision_score(y_C, y_C_proba)
        self.prevalence = float(y_C.mean())

        j = int(np.searchsorted(self.pr_thr, self.opt_tau))
        self.op_prec, self.op_rec = self.prec[j], self.rec[j]

        y_C_proba = self.best_est.predict_proba(X_C)[:, 1]
        y_C_pred = (y_C_proba >= self.opt_tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_C, y_C_pred).ravel()

        return {'prevalence': y_C.mean(), 
                'tau': self.opt_tau, 
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                'beta_C': fn/(fn+tp), 
                'gamma_C': fp/(fp+tn),
                'roc_auc': self.auc, 
                'avg_prec': average_precision_score(y_C, y_C_proba)
        }

    def plot_roc(self, ax):
        ax.plot(self.fpr, self.tpr, label=f'ROC curve (AUC = {self.auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Byzantine Classifier ROC Curve')
        ax.legend(loc='lower right')

    def plot_pr(self,ax):
        ax.plot(self.rec, self.prec, label=f'PR curve (AP = {self.ap:.3f})', linewidth=2)
        ax.axhline(self.prevalence, color='k', linestyle='--', label=f'Random classifier ({self.prevalence:.3f})')
        ax.scatter([self.op_rec], [self.op_prec], color='red', zorder=5, label=f'Operating point (τ = {self.opt_tau:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (1 − FNR)')
        ax.set_ylabel('Precision')
        ax.set_title('Byzantine Classifier PR Curve')
        ax.legend(loc='upper right')

    def log_results(self, res, run_dir, run_id):
        file_path = os.path.join(run_dir, "results.csv")
        res['id'] = run_id

        if not os.path.exists(file_path):
            df_out = pd.DataFrame([res])
            df_out.set_index('id', inplace=True)
            df_out.to_csv(file_path)
        else:
            df_res = pd.DataFrame()
            df_res = pd.read_csv(file_path, index_col='id')
            df_res.loc[run_id] = res 
            df_res.to_csv(file_path)

        return res


