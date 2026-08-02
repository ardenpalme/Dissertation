import threading
import numpy as np
import pandas as pd
from utils import rng, sgd_grad, proj_tau, dirichlet_partition, get_alphas
from metrics import effective_mixing, calc_opt_k
from classifier import MatrixSummaryFeatures
from dist_alg import aggregate, byz_atk

class SystemSimulator():
    def __init__(self, config, global_dataset, rng):
        self.num_nodes = config['sys']['num_nodes']
        self.b = config['sys']['b']
        self.K = config['sys']['K']
        self.batch_sz = config['batch_sz']
        self.rng = rng

        self.X_train = global_dataset['X_train']
        self.y_train = global_dataset['y_train']
        self.X_test = global_dataset['X_test']
        self.y_test = global_dataset['y_test']
        self.C = global_dataset['num_classes']

        self.reg_param = config['reg_param']
        self.print_freq = self.K // 4

    def configure(self, gf, config, proj_const, gamma_C, pre_pipe, best_est, tau_C):
        self.dp = dirichlet_partition(self.y_train, self.num_nodes, config['data_heterogeneity'], self.rng)
        self.G, self.W, sampling_subset = gf.create_graph(config['graph_type'], config['graph_weights'], **config['graph_args'])
        self.B = self.rng.choice(sampling_subset, size=self.b, replace=False)
        self.H = np.array(list(set(np.arange(self.num_nodes)) - set(self.B)))
        _, self.pi, _ = effective_mixing(self.W, self.H, gamma_C)
        self.alphas = get_alphas(self.K, config)
        self.taus = proj_const * self.alphas
        self.pre_pipe = pre_pipe
        self.best_est = best_est
        self.tau_C = tau_C

        # Shared Objects
        self.tar_node = self.rng.choice(self.H)
        self.barrier = threading.Barrier(self.num_nodes) 
        self.models = np.zeros((self.num_nodes, self.X_train.shape[1], self.C))
        self.int_models = np.zeros((self.num_nodes, self.X_train.shape[1], self.C))


    def get_metrics(self, i, models):
        F_k, F_pi_k = calc_opt_k(models[i], self.dp, self.X_train, self.y_train, self.C, self.H, self.pi, self.reg_param)
        return {
            "test_acc": ((self.X_test  @ models[self.H].mean(axis=0)).argmax(1) == self.y_test).mean(),
            "test_acc_pi_obj": ((self.X_test @ np.average(models[self.H], axis=0, weights=self.pi)).argmax(1) == self.y_test).mean(),
            "train_loss": F_k,
            "train_loss_pi": F_pi_k
        }

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
            nbor_byz_filt = y_C_pred_proba > self.tau_C
            
            w_row = self.W[i].copy()
            w_row[nbors[nbor_byz_filt]] = 0.0
            w_row[i] = 1.0 - w_row[nbors].sum()
            models[i] = sum(w_row[j] * proj_tau(models[i], int_models[j], self.taus[k]) for j in np.append(nbors, i))
            barrier.wait()

            if(i == tar_node):
                metrics = self.get_metrics(i, models)
                if(k % self.print_freq == 0 and k>0):
                    print(f"[k={k}] train_loss={metrics['train_loss']:.4f}")
                D = models[self.H] - np.tensordot(self.pi, models[self.H], axes=1)
                metrics['Ck'] = float(self.pi @ (D**2).sum(axis=(1, 2)))
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
                if(k % self.print_freq == 0 and k>0):
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
                if(k % self.print_freq == 0 and k>0):
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
                                         args=(i, tar_node, tar_metrics, models, int_models, self.alphas, rng('sys','sgd',i))) 
                        for i in node_ids]

            return [threading.Thread(target=self.worker_AGG, 
                                     args=(i, barrier, models, int_models, tar_node, tar_metrics[alg], self.alphas, alg, rng('sys', 'sgd', i)))
                    for i in node_ids]
        elif(alg_type == 'byz'):
            return [threading.Thread(target=byz_atk, 
                        args=(i, barrier, models, int_models,  
                              self.X_train[self.dp[i]], self.y_train[self.dp[i]], self.C, self.K, self.W, self.G, 
                              self.reg_param, self.batch_sz, self.alphas, alg, rng('sys','byz', i)))
                       for i in self.B]

        else: return []


    def simulate(self, sim_params):
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
                print(alg.center(len("=" * 25), '='))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        return pd.concat(
            {alg:pd.DataFrame(metric) for [alg,metric] in tar_metrics.items()},
            names=["alg", "k"],
        )

    def get_sim_config(self):
        return (self.W, self.H, self.B, )



