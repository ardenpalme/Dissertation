import os
import threading
import numpy as np
import pandas as pd
from scipy.special import log_softmax 

from utils import parse_results, rng, sgd_grad, proj_tau, dirichlet_partition, get_alphas
from metrics import MetricsCalculator, effective_mixing
from preprocessor import MatrixSummaryFeatures
from dist_alg import aggregate, byz_atk

class SystemSimulator():
    def __init__(self, config, global_dataset, rng, log_fname):
        self.num_nodes = config['sys']['num_nodes']
        self.b = config['sys']['b']
        self.K = config['sys']['K']
        self.batch_sz = config['batch_sz']
        self.rng = rng
        self.log_fname = log_fname

        self.X_train = global_dataset['X_train']
        self.y_train = global_dataset['y_train']
        self.X_test = global_dataset['X_test']
        self.y_test = global_dataset['y_test']
        self.C = global_dataset['num_classes']

        self.reg_param = config['reg_param']
        self.print_freq = self.K // 4

        self.mc = MetricsCalculator(config, global_dataset, rng)
        self.config = config

    def configure(self, gf, config, proj_const, pre_pipe, best_est, params_C, is_printing_logs=True, taus=None):
        self.dp = dirichlet_partition(self.y_train, self.num_nodes, config['data_heterogeneity'], self.rng)
        self.G, self.W, sampling_subset = gf.create_graph(config['graph_type'], config['graph_weights'], **config['graph_args'])
        self.B = self.rng.choice(sampling_subset, size=self.b, replace=False)
        self.H = np.array(list(set(np.arange(self.num_nodes)) - set(self.B)))
        _, self.pi, _ = effective_mixing(self.W, self.H, params_C['C_fpr'])
        self.alphas = get_alphas(self.K, config)
        self.proj_const = proj_const
        self.pre_pipe = pre_pipe
        self.best_est = best_est
        self.params_C = params_C
        self.is_printing_logs = is_printing_logs

        if(taus is None): self.taus = self.proj_const * self.alphas
        else: self.taus = taus

        # Shared Objects
        self.tar_node = self.rng.choice(self.H)
        self.barrier = threading.Barrier(self.num_nodes) 
        self.models = np.zeros((self.num_nodes, self.X_train.shape[1], self.C))
        self.int_models = np.zeros((self.num_nodes, self.X_train.shape[1], self.C))
        self.test_acc_arr = np.zeros(self.num_nodes)
        self.E_sq = np.zeros(self.num_nodes)
        self.test_loss_arr = np.zeros(self.num_nodes)

    @staticmethod
    def calc_opt_k(theta_ref, dp, X, y, C, H, pi, reg_lambda):
        F_Si_arr = np.zeros(len(H))
        for i in range(len(H)):
            node_idx = H[i]
            Xl, yl = X[dp[node_idx]], y[dp[node_idx]]
            ce_loss = -(np.eye(C, dtype=int)[yl] * log_softmax(Xl @ theta_ref, axis=1)).sum(axis=1)
            F_Si_arr[i] = ce_loss.mean() + (0.5 * reg_lambda * (np.linalg.norm(theta_ref)**2))
        return F_Si_arr.mean(), np.average(F_Si_arr, weights=pi)

    def get_metrics(self, i, models):
        F_k, F_pi_k = self.calc_opt_k(models[i], self.dp, self.X_train, self.y_train, self.C, self.H, self.pi, self.reg_param)
        return {
            "test_acc": ((self.X_test  @ models[self.H].mean(axis=0)).argmax(1) == self.y_test).mean(),
            "test_acc_pi_obj": ((self.X_test @ np.average(models[self.H], axis=0, weights=self.pi)).argmax(1) == self.y_test).mean(),
            "train_loss": F_k,
            "train_loss_pi": F_pi_k,
        }

    def worker_RDSGD_oracle(self, i, barrier, models, int_models, tar_node, tar_metrics, alphas, rng):
        Xl, yl = self.X_train[self.dp[i]], self.y_train[self.dp[i]]
        nbors = np.array(list(self.G.neighbors(i)))
        byz_nbors = np.isin(nbors, self.B)
        hon_nbors = np.isin(nbors, self.H)

        for k in range(self.K):
            _, g = sgd_grad(Xl, yl, models[i], self.reg_param, self.batch_sz, rng)
            int_models[i] = models[i] - alphas[k] * g
            barrier.wait()
            
            drop = np.where((byz_nbors & (rng.random(len(nbors)) < (1-self.params_C['C_fnr']))) | \
                    (hon_nbors & (rng.random(len(nbors)) < self.params_C['C_fpr'])))
            w_row = self.W[i].copy()
            w_row[np.asarray(nbors)[drop]] = 0.0
            w_row[i] = 1.0 - w_row[nbors].sum()
            models[i] = sum(w_row[j] * proj_tau(models[i], int_models[j], self.taus[k]) for j in np.append(nbors, i))

            self.test_acc_arr[i] = ((self.X_test @ models[self.H].mean(axis=0)).argmax(1) == self.y_test).mean()

            ce_loss = -(np.eye(self.C, dtype=int)[self.y_test] * log_softmax(self.X_test @ models[i], axis=1)).sum(axis=1)
            self.test_loss_arr[i] = ce_loss.mean() + (0.5 * self.reg_param * (np.linalg.norm(models[i])**2))
            barrier.wait()

            if(i == tar_node):
                metrics = self.get_metrics(i, models)
                metrics['min_test_acc'] = np.min(self.test_acc_arr[self.H])
                metrics['max_test_acc'] = np.max(self.test_acc_arr[self.H])
                if(k % self.print_freq == 0 and k>0 and self.is_printing_logs):
                    print(f"[k={k}] train_loss={metrics['train_loss']:.4f}")
                D = models[self.H] - np.tensordot(self.pi, models[self.H], axes=1)
                metrics['Ck'] = float(self.pi @ (D**2).sum(axis=(1, 2)))
                metrics['min_test_loss'] = self.test_loss_arr[self.H].min()
                metrics['max_test_loss'] = self.test_loss_arr[self.H].max()
                tar_metrics.append(metrics)        

    def worker_RDSGD(self, i, barrier, models, int_models, tar_node, tar_metrics, alphas, rng):
        Xl, yl = self.X_train[self.dp[i]], self.y_train[self.dp[i]]
        feat = MatrixSummaryFeatures(i, Xl, yl, alphas)
        nbors = np.array(list(self.G.neighbors(i)))
        
        for k in range(self.K):
            _, g = sgd_grad(Xl, yl, models[i], self.reg_param, self.batch_sz, rng)
            int_models[i] = models[i] - alphas[k] * g
            barrier.wait()
            
            feat.set_context(k, models[i], g)
            Z = feat.transform(np.stack([int_models[j] for j in nbors]))
            nbor_int_models = self.pre_pipe.transform(Z)
            y_C_pred_proba = self.best_est.predict_proba(nbor_int_models)[:,1]
            nbor_byz_filt = y_C_pred_proba > self.params_C['C_tau']
            
            w_row = self.W[i].copy()
            w_row[nbors[nbor_byz_filt]] = 0.0
            w_row[i] = 1.0 - w_row[nbors].sum()

            x_prev = models[i].copy()
            proj = {j: proj_tau(x_prev, int_models[j], self.taus[k]) for j in nbors}

            models[i] = w_row[i] * x_prev + sum(w_row[j] * proj[j] for j in nbors)

            E_i = sum((w_row[j] * (proj[j] - x_prev) for j in nbors if j in self.B and w_row[j] != 0.0), 
                      start=np.zeros_like(x_prev))
            self.E_sq[i] = float(np.sum(E_i**2))

            ce_loss = -(np.eye(self.C, dtype=int)[self.y_test] * log_softmax(self.X_test @ models[i], axis=1)).sum(axis=1)
            self.test_loss_arr[i] = ce_loss.mean() + (0.5 * self.reg_param * (np.linalg.norm(models[i])**2))
            barrier.wait()

            if(i == tar_node):
                metrics = self.get_metrics(i, models)
                if(k % self.print_freq == 0 and k>0 and self.is_printing_logs):
                    print(f"[k={k}] train_loss={metrics['train_loss']:.4f}")
                D = models[self.H] - np.tensordot(self.pi, models[self.H], axes=1)
                metrics['Ck'] = float(self.pi @ (D**2).sum(axis=(1, 2)))
                metrics['rho2_k'] = float(self.pi @ self.E_sq[self.H])
                metrics['min_test_loss'] = self.test_loss_arr[self.H].min()
                metrics['max_test_loss'] = self.test_loss_arr[self.H].max()
                tar_metrics.append(metrics)        

    def worker_AGG(self, i, barrier, models, int_models, tar_node, tar_metrics, alphas, agg_rule, rng):
        Xl, yl = self.X_train[self.dp[i]], self.y_train[self.dp[i]]
        for k in range(self.K):
            _, g = sgd_grad(Xl, yl, models[i], self.reg_param, self.batch_sz, rng)
            int_models[i] = models[i] - alphas[k] * g 
            barrier.wait()
            models[i] = aggregate(i, int_models, self.W, self.G, self.B, self.H, agg_rule)
            barrier.wait()
            if(i == tar_node):
                metrics = self.get_metrics(i, models)
                if(k % self.print_freq == 0 and k>0 and self.is_printing_logs):
                    print(f"[k={k}] train_loss={metrics['train_loss']:.4f}")
                tar_metrics.append(metrics)


    def worker_DSGD(self, i, barrier, tar_node, tar_metrics, models, int_models, alphas, rng):
        Xl, yl = self.X_train[self.dp[i]], self.y_train[self.dp[i]]
        for k in range(self.K):
            _, g = sgd_grad(Xl, yl, models[i], self.reg_param, self.batch_sz, rng)
            int_models[i] = models[i] - alphas[k] * g
            barrier.wait()       
            models[i] = sum(self.W[i, j] * int_models[j] for j in list(self.G.neighbors(i)) + [i])
            barrier.wait()
                
            if(i == tar_node):
                metrics = self.get_metrics(i, models)
                if(k % self.print_freq == 0 and k>0 and self.is_printing_logs):
                    print(f"[k={k}] train_loss={metrics['train_loss']:.4f}")
                tar_metrics.append(metrics)


    def create_threads(self, config_params, barrier, models, int_models, tar_metrics, tar_node):
        alg_type, alg, node_ids = config_params
        if(alg_type == 'hon'):
            if(alg == 'RDSGD'):
                return [threading.Thread(target=self.worker_RDSGD, 
                                         args=(i, barrier, models, int_models, tar_node, tar_metrics['RDSGD'],
                                               self.alphas, rng('sys','sgd',i)))
                        for i in node_ids]
            elif(alg == 'DSGD'):
                return [threading.Thread(target=self.worker_DSGD, 
                                         args=(i, tar_node, tar_metrics['DSGD'], models, int_models, self.alphas, rng('sys','sgd',i))) 
                        for i in node_ids]
            elif(alg == 'RDSGD_ORACLE'):
                return [threading.Thread(target=self.worker_RDSGD_oracle, 
                                         args=(i, barrier, models, int_models, tar_node, tar_metrics['RDSGD_ORACLE'],
                                               self.alphas, rng('sys','sgd',i)))
                        for i in node_ids]

            return [threading.Thread(target=self.worker_AGG, 
                                     args=(i, barrier, models, int_models, tar_node, tar_metrics[alg], self.alphas, alg, rng('sys', 'sgd', i)))
                    for i in node_ids]
        elif(alg_type == 'byz'):
            return [threading.Thread(target=byz_atk, 
                args=(i, self.barrier, self.models, self.int_models, self.X_train, self.y_train, self.dp, 
                      self.C, self.K, self.W, self.G, self.H,
                      self.reg_param, self.batch_sz, self.alphas, self.taus, alg, rng('sys', 'byz', i)),
                kwargs={'alie_z': None, 'num_nodes':self.num_nodes, 'b': self.b,
                        'ipm_eps':0.5})
               for i in self.B]
        else: return []


    def simulate(self, sim_params, run_results):
        algorithms = sim_params['algorithms']
        atk_type = sim_params['atk_type']
        threat_model = sim_params['threat_model']

        tar_metrics = {alg : list() for alg in algorithms}
        if(threat_model == 'T0'):
            for alg in algorithms:
                self.models.fill(0)
                self.int_models.fill(0)
                threads = self.create_threads(('hon', alg, np.arange(self.num_nodes)), 
                                              self.barrier, self.models, self.int_models, tar_metrics, self.tar_node)
                if self.is_printing_logs:
                    print(alg.center(len("=" * 25), '='))

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        elif(threat_model == 'T3'):
            for alg in algorithms:
                self.models.fill(0)
                self.int_models.fill(0)
                hon_threads = self.create_threads(('hon', alg, self.H), 
                                              self.barrier, self.models, self.int_models, tar_metrics, self.tar_node)

                byz_threads = self.create_threads(('byz', atk_type, self.B), 
                                              self.barrier, self.models, self.int_models, tar_metrics, self.tar_node)
                threads = hon_threads + byz_threads 

                if self.is_printing_logs:
                    print(alg.center(len("=" * 25), '='))

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        df = pd.concat(
            {alg:pd.DataFrame(metric) for [alg,metric] in tar_metrics.items()},
            names=["alg", "k"],
        )

        for alg in algorithms:
            run_results[f'{threat_model}_{alg}'] = dict(test_acc=df.loc[alg]['test_acc'].iloc[-1])

        return df

    def log_results(self, res, run_dir, run_id):
        sim_config = (self.W, self.H, self.B, self.dp, self.pi)
        file_path = os.path.join(run_dir, "results.csv")

        data = self.config.copy()
        data.update(res)
        data.update(self.mc(sim_config, self.models, self.params_C['C_fpr'], self.params_C['C_fnr'], self.proj_const))
        payload = parse_results(data)
        payload['id'] = run_id

        if not os.path.exists(file_path):
            df_out = pd.DataFrame([payload])
            df_out.set_index('id', inplace=True)
            df_out.to_csv(file_path)
        else:
            df_res = pd.DataFrame()
            df_res = pd.read_csv(file_path, index_col='id')
            df_res.loc[run_id] = payload 
            df_res.to_csv(file_path)

        return payload

