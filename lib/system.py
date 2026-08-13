import threading
import numpy as np
import pandas as pd
from scipy.special import log_softmax

from utils import rng, sgd_grad, proj_tau, dirichlet_partition, get_alphas
from metrics import MetricsCalculator, effective_mixing
from preprocessor import FeaturesTransformer
from dist_alg import aggregate, byz_atk

class SystemSimulator():
    def __init__(self, config, global_dataset, gf):
        self.num_nodes = config['sys']['num_nodes']
        self.b = config['sys']['b']
        self.K = config['sys']['K']
        self.batch_sz = config['batch_sz']

        self.gf = gf
        self.mc = MetricsCalculator(config, global_dataset, rng)

        self.X_train = global_dataset['X_train']
        self.y_train = global_dataset['y_train']
        self.X_test = global_dataset['X_test']
        self.y_test = global_dataset['y_test']
        self.C = global_dataset['num_classes']

        self.reg_param = config['reg_param']
        self.print_freq = self.K // 4


    def init_simulation(self, config, proj_const, pre_pipe, best_est, oracle_params, rdsgd_params, seed, is_printing_logs=True, taus=None):
        self.rng = rng(seed)
        self.dp = dirichlet_partition(self.y_train, self.num_nodes, config['data_heterogeneity'], self.rng)
        self.G, self.W, self.B, self.H = self.gf.create_graph(config['graph_type'], config['graph_weights'], seed, **config['graph_args'])
        self.Wbar, self.pi, self.lam_pi = effective_mixing(self.W, self.H, rdsgd_params['C_fpr'], config['graph_args']['MH_target_pi'])
        self.alphas = get_alphas(self.K, config)
        self.proj_const = proj_const
        self.pre_pipe = pre_pipe
        self.clf = best_est
        self.oracle_params = oracle_params
        self.rdsgd_params = rdsgd_params 

        self.X_shards = [self.X_train[self.dp[i]] for i in self.H]
        self.X_H = np.vstack(self.X_shards)
        self.y_H = np.concatenate([self.y_train[self.dp[i]] for i in self.H])

        local_opt_res = self.mc.calc_local_opt(self.X_H, self.y_H, self.H, self.pi, self.dp, self.reg_param)
        self.x_opt = local_opt_res['x_star']
        self.x_pi_opt = local_opt_res['x_pi_star']
        self.zeta_sq_pi = self.mc.calc_zeta_sq_pi(self.x_pi_opt, self.X_train, self.y_train, self.dp, self.H, self.pi, self.C)
        self.sigma_sq = self.mc.calc_sigma_sq(self.x_opt, self.X_train, self.y_train, self.dp, self.H, self.C)
        self.dB_max = max(self.W[i, self.B].sum() for i in self.H) if len(self.B) else 0.0

        self.is_printing_logs = is_printing_logs
        self.taus = taus if taus is not None else (self.proj_const * self.alphas)

        # Shared Objects
        self.tar_node = self.rng.choice(self.H)
        self.barrier = threading.Barrier(self.num_nodes) 
        self.models = np.zeros((self.num_nodes, self.X_train.shape[1], self.C))
        self.int_models = np.zeros((self.num_nodes, self.X_train.shape[1], self.C))
        self.test_acc_arr = np.zeros(self.num_nodes)
        self.E_sq = np.zeros(self.num_nodes)
        self.train_loss_arr = np.zeros(self.num_nodes)
        self.edge_log = np.zeros((self.num_nodes, self.K, 8))

    def calc_reduced_rdsgd_consts(self):
        L, mu = self.mc.calc_L_mu(self.models[self.H], self.X_shards, self.reg_param)
        return dict(L=L, mu=mu, zeta_sq_pi=self.zeta_sq_pi, sigma_sq=self.sigma_sq, dB_max=self.dB_max)

    @staticmethod
    def calc_opt_k(theta_ref, dp, X, y, C, H, pi, reg_lambda):
        F_Si_arr = np.zeros(len(H))
        for i in range(len(H)):
            node_idx = H[i]
            Xl, yl = X[dp[node_idx]], y[dp[node_idx]]
            ce_loss = -(np.eye(C, dtype=int)[yl] * log_softmax(Xl @ theta_ref, axis=1)).sum(axis=1)
            F_Si_arr[i] = ce_loss.mean() + (0.5 * reg_lambda * (np.linalg.norm(theta_ref)**2))
        return F_Si_arr.mean(), np.average(F_Si_arr, weights=pi)

    def get_metrics_summary(self, models):
        h = len(self.H)
        Xh = models[self.H]
        Pi = np.eye(h) - np.outer(np.ones(h),self.pi)
        X_perp = np.tensordot(Pi, Xh, axes=(1, 0))
        unif_dist=np.full(h,1/h)
        x_pi_k = np.tensordot(self.pi, Xh, axes=(0,0))

        return {
            'test_acc':       ((self.X_test @ Xh.mean(axis=0)).argmax(1) == self.y_test).mean(),
            'test_acc_pi':    ((self.X_test @ np.tensordot(self.pi, Xh, axes=1)).argmax(1) == self.y_test).mean(),
            'C_pi':           np.einsum('ijk,i->', X_perp**2, self.pi),
            'C_unif' :        np.einsum('ijk,i->', X_perp**2, unif_dist),
            'mean_test_acc':  self.test_acc_arr[self.H].mean(),
            'min_test_acc':   self.test_acc_arr[self.H].min(),
            'max_test_acc':   self.test_acc_arr[self.H].max(),
            'min_train_loss': self.train_loss_arr[self.H].min(),
            'max_train_loss': self.train_loss_arr[self.H].max(),
            'opt_gap_pi':     np.linalg.norm(x_pi_k - self.x_pi_opt),
            'opt_gap':        np.linalg.norm(x_pi_k - self.x_opt),
        }

    def calc_realized_fpr_fnr(self,k):
        eps = 10**(-12)
        E = self.edge_log[self.H]
        tp, fp, tn, fn = E[:,k, 0], E[:,k,1], E[:,k,2], E[:,k,3]
        gamma_k = fp.sum() / (fp.sum() + tn.sum() + eps)
        beta_k = fn.sum() / (fn.sum() + tp.sum() + eps)
        byz_w_k = E[:,k,7] @ self.pi

        fp_i, tn_i = self.edge_log[self.H][:, k, 1], self.edge_log[self.H][:, k, 2]
        gamma_k_node = fp_i / (fp_i + tn_i + eps) 

        return dict(gamma_k=gamma_k, beta_k=beta_k, gamma_k_node=gamma_k_node, byz_w_k=byz_w_k)

    def calc_reduced_local_metrics(self, i, models): # ORACLE and Aggregators
        Xl, yl = self.X_train[self.dp[i]], self.y_train[self.dp[i]]

        # test_acc_k
        self.test_acc_arr[i] = ((self.X_test @ models[i]).argmax(1) == self.y_test).mean()

        # train_loss_k 
        ce = -(np.eye(self.C, dtype=int)[yl] * log_softmax(Xl @ models[i], axis=1)).sum(axis=1)
        self.train_loss_arr[i] = ce.mean() + 0.5 * self.reg_param * np.linalg.norm(models[i])**2


    def calc_full_local_metrics(self, i, k, models, w_row, proj, x_prev, nbors, flagged, is_adv, is_hon):
        self.calc_reduced_local_metrics(i, models)

        # rho2_k 
        E_i = sum((w_row[j] * (proj[j] - x_prev) for j in nbors if j in self.B and w_row[j] != 0.0), 
                  start=np.zeros_like(x_prev))
        self.E_sq[i] = float(np.sum(E_i**2))

        # fpr_k and fnr_k 
        tp, fn = flagged & is_adv,  ~flagged & is_adv
        fp, tn = flagged & is_hon,  ~flagged & is_hon
        self.edge_log[i, k] = [tp.sum(), fp.sum(), tn.sum(), fn.sum(),
                               self.W[i, nbors][tp].sum(), self.W[i, nbors][fp].sum(), 
                               self.W[i, nbors][tn].sum(), self.W[i, nbors][fn].sum()]

    def worker_RDSGD_oracle(self, i, barrier, models, int_models, tar_node, tar_metrics, alphas, rng, oracle_id):
        Xl, yl = self.X_train[self.dp[i]], self.y_train[self.dp[i]]
        nbors = np.array(list(self.G.neighbors(i)))
        is_byz  = np.isin(nbors, self.B)
        is_adv  = is_byz & ~self.abstain[nbors]
        is_hon  = ~is_byz

        for k in range(self.K):
            _, g = sgd_grad(Xl, yl, models[i], self.reg_param, self.batch_sz, rng)
            int_models[i] = models[i] - alphas[k] * g
            barrier.wait() # all intermediate models (x^{k+1/2}) written
            barrier.wait() # Byzantine x^{k+1/2} written
            
            flagged = (is_adv & (rng.random(len(nbors)) < (1-self.oracle_params[oracle_id]['C_fnr'][i]))) | \
                    (is_hon & (rng.random(len(nbors)) < self.oracle_params[oracle_id]['C_fpr'][i]))

            w_row = self.W[i].copy()
            w_row[nbors[flagged]] = 0.0
            w_row[i] = 1.0 - w_row[nbors].sum()

            proj = {j: proj_tau(models[i], int_models[j], self.taus[k]) for j in np.append(nbors, i)}
            models[i] = sum(w_row[j] * proj[j] for j in np.append(nbors, i)) 
            barrier.wait() # all models (x^{k+1}) written

            # fpr_k and fnr_k 
            tp, fn = flagged & is_adv,  ~flagged & is_adv
            fp, tn = flagged & is_hon,  ~flagged & is_hon
            self.edge_log[i, k] = [tp.sum(), fp.sum(), tn.sum(), fn.sum(),
                                   self.W[i, nbors][tp].sum(), self.W[i, nbors][fp].sum(), 
                                   self.W[i, nbors][tn].sum(), self.W[i, nbors][fn].sum()]

            self.calc_reduced_local_metrics(i, models)
            barrier.wait() # all metrics collected by all honest workers

            if(i == tar_node):
                metrics = self.get_metrics_summary(models)
                metrics.update(self.calc_realized_fpr_fnr(k))
                if(k % self.print_freq == 0 and k>0 and self.is_printing_logs):
                    print(f"[k={k}] "
                          f"train_loss: [{metrics['min_train_loss']:.3f},{metrics['max_train_loss']:.3f}],"
                          f"test_acc: [{metrics['min_test_acc']:.3f},{metrics['max_test_acc']:.3f}]")

                tar_metrics.append(metrics)        

    def worker_RDSGD(self, i, barrier, models, int_models, tar_node, tar_metrics, alphas, rng):
        Xl, yl = self.X_train[self.dp[i]], self.y_train[self.dp[i]]
        feat = FeaturesTransformer(i, Xl, yl, alphas, self.reg_param)
        nbors = np.array(list(self.G.neighbors(i)))
        is_byz  = np.isin(nbors, self.B)
        is_adv  = is_byz & ~self.abstain[nbors]
        is_hon  = ~is_byz

        for k in range(self.K):
            _, g = sgd_grad(Xl, yl, models[i], self.reg_param, self.batch_sz, rng)
            int_models[i] = models[i] - alphas[k] * g
            barrier.wait() # all intermediate models (x^{k+1/2}) written
            barrier.wait() # Byzantine x^{k+1/2} written
            
            feat.set_context(k, models[i], g)
            Z = self.pre_pipe.transform(feat.transform(np.stack([int_models[j] for j in nbors])))
            probs = self.clf.predict_proba(Z)[:, 1]

            flagged = (probs >= self.rdsgd_params['C_tau'][i])

            w_row = self.W[i].copy()
            w_row[nbors[flagged]] = 0.0
            w_row[i] = 1.0 - w_row[nbors].sum()

            x_prev = models[i].copy()
            proj = {j: proj_tau(models[i], int_models[j], self.taus[k]) for j in np.append(nbors, i)}
            models[i] = sum(w_row[j] * proj[j] for j in np.append(nbors, i)) 
            barrier.wait() # all models (x^{k+1}) written

            self.calc_full_local_metrics(i, k, models, w_row, proj, x_prev, nbors, flagged, is_adv, is_hon)
            barrier.wait() # all metrics collected by all honest workers

            if(i == tar_node):
                metrics = self.get_metrics_summary(models)
                metrics['rho2_k'] = float(self.pi @ self.E_sq[self.H])
                metrics.update(self.calc_realized_fpr_fnr(k))
                if(k % self.print_freq == 0 and k>0 and self.is_printing_logs):
                    print(f"[k={k}] "
                          f"train_loss: [{metrics['min_train_loss']:.3f},{metrics['max_train_loss']:.3f}],"
                          f"test_acc: [{metrics['min_test_acc']:.3f},{metrics['max_test_acc']:.3f}]")
                tar_metrics.append(metrics)        

    def worker_AGG(self, i, barrier, models, int_models, tar_node, tar_metrics, alphas, agg_rule, rng):
        Xl, yl = self.X_train[self.dp[i]], self.y_train[self.dp[i]]
        for k in range(self.K):
            _, g = sgd_grad(Xl, yl, models[i], self.reg_param, self.batch_sz, rng)
            int_models[i] = models[i] - alphas[k] * g 
            barrier.wait() # all intermediate models (x^{k+1/2}) written
            barrier.wait() # Byzantine x^{k+1/2} written

            models[i] = aggregate(i, int_models, self.W, self.G, self.B, self.H, agg_rule)
            barrier.wait() # all models (x^{k+1}) written

            self.calc_reduced_local_metrics(i, models)
            barrier.wait() # all metrics collected by all honest workers

            if(i == tar_node):
                metrics = self.get_metrics_summary(models)
                if(k % self.print_freq == 0 and k>0 and self.is_printing_logs):
                    print(f"[k={k}] "
                          f"train_loss: [{metrics['min_train_loss']:.3f},{metrics['max_train_loss']:.3f}],"
                          f"test_acc: [{metrics['min_test_acc']:.3f},{metrics['max_test_acc']:.3f}]")
                tar_metrics.append(metrics)        


    def worker_DSGD(self, i, barrier, models, int_models, tar_node, tar_metrics, alphas, rng):
        Xl, yl = self.X_train[self.dp[i]], self.y_train[self.dp[i]]
        for k in range(self.K):
            _, g = sgd_grad(Xl, yl, models[i], self.reg_param, self.batch_sz, rng)
            int_models[i] = models[i] - alphas[k] * g
            barrier.wait() # all intermediate models (x^{k+1/2}) written

            models[i] = sum(self.W[i, j] * int_models[j] for j in list(self.G.neighbors(i)) + [i])
            barrier.wait() # all models (x^{k+1}) written

            self.calc_reduced_local_metrics(i, models)
            barrier.wait() # all metrics collected by all honest workers
                
            if(i == tar_node):
                metrics = self.get_metrics_summary(models)
                if(k % self.print_freq == 0 and k>0 and self.is_printing_logs):
                    print(f"[k={k}] "
                          f"train_loss: [{metrics['min_train_loss']:.3f},{metrics['max_train_loss']:.3f}],"
                          f"test_acc: [{metrics['min_test_acc']:.3f},{metrics['max_test_acc']:.3f}]")
                tar_metrics.append(metrics)        


    def create_threads(self, config_params, barrier, models, int_models, tar_metrics, tar_node, seed):
        alg_type, alg, node_ids = config_params
        if(alg_type == 'hon'):
            if(alg == 'RDSGD'):
                return [threading.Thread(target=self.worker_RDSGD, 
                                         args=(i, barrier, models, int_models, tar_node, tar_metrics['RDSGD'],
                                               self.alphas, rng('sys','sgd', seed, i)))
                        for i in node_ids]
            elif(alg == 'DSGD'):
                return [threading.Thread(target=self.worker_DSGD, 
                                         args=(i, barrier,  models, int_models, tar_node, tar_metrics['DSGD'],
                                               self.alphas, rng('sys','sgd', seed, i))) 
                        for i in node_ids]
            elif(alg.startswith('ORACLE')):
                oracle_id = int(alg.split("_")[1])
                return [threading.Thread(target=self.worker_RDSGD_oracle, 
                                         args=(i, barrier, models, int_models, tar_node, tar_metrics[alg],
                                               self.alphas, rng('sys','sgd', seed, i), oracle_id))
                        for i in node_ids]

            return [threading.Thread(target=self.worker_AGG, 
                                     args=(i, barrier, models, int_models, tar_node, tar_metrics[alg], 
                                           self.alphas, alg, rng('sys', 'sgd', seed, i)))
                    for i in node_ids]
        elif(alg_type == 'byz'):
            return [threading.Thread(target=byz_atk, 
                args=(i, self.barrier, self.models, self.int_models, self.X_train, self.y_train, self.dp, 
                      self.C, self.K, self.W, self.G, self.H,
                      self.reg_param, self.batch_sz, self.alphas, alg, rng('sys', 'byz', seed, i)),
                kwargs={'ipm_eps':0.5, 'abstain': self.abstain})
               for i in self.B]
        else: return []

    def _abstain_mask(self, atk_type):
        a = np.zeros(self.num_nodes, dtype=bool)
        if atk_type not in ('ALIE', 'IPM'):
            return a
        min_hon = 2 if atk_type == 'ALIE' else 1
        H_set = set(int(j) for j in self.H)
        for i in self.B:
            n_hon = sum(int(j) in H_set for j in self.G.neighbors(int(i)))
            a[int(i)] = n_hon < min_hon
        return a


    def simulate(self, sim_params):
        algorithms = sim_params['algorithms']
        atk_type = sim_params['atk_type']
        threat_model = sim_params['threat_model']
        seed = sim_params['seed']

        self.abstain = self._abstain_mask(atk_type)
        self.n_abstain = int(self.abstain.sum())

        tar_metrics = {alg : list() for alg in algorithms}
        if(threat_model == 'T0'):
            for alg in algorithms:
                self.models.fill(0)
                self.int_models.fill(0)
                threads = self.create_threads(('hon', alg, np.arange(self.num_nodes)), 
                                              self.barrier, self.models, self.int_models, tar_metrics, self.tar_node, seed)
                if self.is_printing_logs:
                    print(alg.center(len("=" * 56), '='))

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        elif(threat_model == 'T3'):
            for alg in algorithms:
                self.models.fill(0)
                self.int_models.fill(0)
                hon_threads = self.create_threads(('hon', alg, self.H), 
                                              self.barrier, self.models, self.int_models, tar_metrics, self.tar_node, seed)

                byz_threads = self.create_threads(('byz', atk_type, self.B), 
                                              self.barrier, self.models, self.int_models, tar_metrics, self.tar_node, seed)
                threads = hon_threads + byz_threads 

                if self.is_printing_logs:
                    print(alg.center(len("=" * 56), '='))

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        df = pd.concat(
            {alg:pd.DataFrame(metric) for [alg,metric] in tar_metrics.items()},
            names=["alg", "k"],
        )

        return df

    def get_sim_config(self):
        return (self.W, self.H, self.B, self.dp, self.pi)

