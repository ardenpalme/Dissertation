import threading
import numpy as np
from utils import rng, sgd_grad, dirichlet_partition, get_alphas

class ProjConstEstimator():
    def __init__(self, config, global_dataset, gf):
        self.num_nodes = config['train']['num_nodes']
        self.b = config['train']['b']
        self.K = config['train']['K']
        self.batch_sz = config['batch_sz']
        self.iter_sample_sz = self.K//2
        self.gf = gf

        self.X_train = global_dataset['X_train']
        self.y_train = global_dataset['y_train']
        self.C = global_dataset['num_classes']

        self.reg_param = config['reg_param']
        self.eta = 1.2

    def configure(self, config, seed):
        self.rng = rng(seed)
        self.dp = dirichlet_partition(self.y_train, self.num_nodes, config['data_heterogeneity'], self.rng)
        self.G, self.W, self.B, self.H = self.gf.create_graph(config['graph_type'], config['graph_weights'], seed, **config['graph_args'])
        self.alphas = get_alphas(self.K, config)

    def worker_DSGD(self, i, barrier, models, int_models, alphas, r_samples, rng):
        Xl, yl = self.X_train[self.dp[i]], self.y_train[self.dp[i]]
        nbors = np.array(list(self.G.neighbors(i)))
        for k in range(self.K):
            _, g = sgd_grad(Xl, yl, models[i], self.reg_param, self.batch_sz, rng)
            int_models[i] = models[i] - alphas[k] * g
            barrier.wait()
            r_samples[i].extend(np.linalg.norm(int_models[j] - int_models[i]) / alphas[k] for j in nbors)
            models[i] = sum(self.W[i, j] * int_models[j] for j in list(self.G.neighbors(i)) + [i])
            barrier.wait()

    def estimate(self):
        barrier = threading.Barrier(self.num_nodes) 
        models = np.zeros((self.num_nodes, self.X_train.shape[1], self.C))
        int_models = np.zeros((self.num_nodes, self.X_train.shape[1], self.C))
        r_samples = [[] for _ in range(self.num_nodes)]

        DSGD_threads = [threading.Thread(target=self.worker_DSGD, 
                                         args=(i, barrier, models, int_models, self.alphas, 
                                               r_samples, rng('proj-const-estimator','sim',i)))
                        for i in np.arange(self.num_nodes)]

        for thread in DSGD_threads:
            thread.start()
        for thread in DSGD_threads:
            thread.join()

        residuals = np.concatenate([np.asarray(s) for i, s in enumerate(r_samples) if i in range(self.num_nodes)])
        proj_const = self.eta * np.quantile(residuals, 0.8)
        return proj_const
