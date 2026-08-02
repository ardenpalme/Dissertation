import zlib
import numpy as np
from scipy.special import softmax, log_softmax 

global_seed = 12345

def seed(*path):
    return zlib.crc32(str((global_seed,) + path).encode())

def rng(*path):
    return np.random.default_rng(seed(*path))


def proj_tau(x_i, x_j, tau):
    d = x_j - x_i
    nd = np.linalg.norm(d)
    return x_i if nd == 0 else x_i + min(1.0, tau / nd) * d

def sgd_grad(X, y, theta, lam, batch_sz, rng):
    n, K = X.shape[0], theta.shape[1]
    idx = rng.choice(n, size=batch_sz, replace=False) 
    Xb, yb = X[idx], y[idx]
    Z  = Xb @ theta
    Yb = np.eye(K)[yb]
    loss = -(Yb * log_softmax(Z, axis=1)).sum() / batch_sz + 0.5 * lam * np.linalg.norm(theta)**2
    grad = Xb.T @ (softmax(Z, axis=1) - Yb) / batch_sz + lam * theta
    return loss, grad

def global_ce(theta, X, y):
    z = X @ theta
    z -= z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -log_probs[np.arange(len(y)), y].mean()

def dirichlet_partition(labels, n_clients, alpha, rng): #TODO understand
    labels = np.asarray(labels)
    m = len(labels) // n_clients                 # equal shard size
    room = np.full(n_clients, m)                 # remaining capacity per client
    client_idx = [[] for _ in range(n_clients)]

    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        w = rng.dirichlet(alpha * np.ones(n_clients))

        pos = 0
        while pos < len(idx_c) and room.sum() > 0:
            p = w * (room > 0)                   # only clients with space left
            p = p / p.sum() if p.sum() else (room > 0) / (room > 0).sum()
            take = np.minimum(rng.multinomial(len(idx_c) - pos, p), room)
            if take.sum() == 0:                  # all mass landed on full clients
                take = np.zeros(n_clients, int)
                take[rng.choice(n_clients, p=p)] = 1
            for i in np.flatnonzero(take):
                client_idx[i].extend(idx_c[pos:pos + take[i]])
                pos += take[i]
                room[i] -= take[i]

    return [np.array(ci) for ci in client_idx]

def get_alphas(K, config):
    return config['alpha_init'] / (1.0 + (np.arange(K)*config['alpha_decay']))

def parse(dat):
    assert dat['train']['num_nodes'] == dat['sys']['num_nodes']
    assert dat['train']['K'] == dat['sys']['K']
    assert dat['train']['b'] == dat['sys']['b']
    
    row = dat.copy()
    row['n'] = dat['sys']['num_nodes']
    row['K'] = dat['sys']['K']
    row['b'] = dat['sys']['b']
    row['sim_gamma_C'] = dat['train']['gamma_C']   # simulation FPR used in Bernoulli dropout (imposed)
    row['sim_beta_C'] = dat['train']['beta_C']     # simulation FNR used in Bernoulli dropout (imposed)
    row['val_gamma_C'] = dat['classifier']['fpr']  # measured FPR (distinct sim testset)
    row['val_beta_C'] = dat['classifier']['fnr']   # measured FNR (distinct sim testset) 
    row['val_auc'] = dat['classifier']['auc']      # measured AUC (distinct sim testset) 
    row['tau_log'] = dat['classifier']['tau']      # ideal logistic reg. threshold tau that minimizes FNR on distinct sim testset
    del row['classifier']

    # Parse run results
    for alg in ['RDSGD', 'IOS', 'SCC', 'TriMean', 'CooMed']:
        row[f'{alg}_T3'] = dat['T3'][alg]['test_acc']
    for alg in ['RDSGD', 'IOS', 'SCC', 'TriMean', 'CooMed', 'DSGD']:
        row[f'{alg}_T0'] = dat['T0'][alg]['test_acc']
    del row['T0']
    del row['T3']
    
    # Parse system fields
    if 'atk_type' in row['sys'].keys():
        row['atk_type'] = row['sys']['atk_type']
    del row['sys']

    # Parse classifier training fields
    train_keys = list(row['train'].keys())
    if ('train_atks' in train_keys) and ('val_atks' in train_keys):
        row['train_atks'] = set(row['train']['train_atks'])
        row['val_atks'] = set(row['train']['val_atks'])
    else:
        row['train_atks'] = {row['atk_type']}
        row['val_atks'] = {row['atk_type']}
    del row['train']
    
    if('x_star' in dat.keys()):
        del row['x_star']

    return row
