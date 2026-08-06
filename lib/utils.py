import zlib
import numpy as np
from scipy.special import softmax, log_softmax 
from sklearn.datasets import fetch_openml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from config import GLOBAL_SEED

def seed(*path):
    return zlib.crc32(str((GLOBAL_SEED,) + path).encode())

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

def dirichlet_partition(labels, n_clients, alpha, rng): 
    labels = np.asarray(labels)
    m = len(labels) // n_clients
    room = np.full(n_clients, m)
    client_idx = [[] for _ in range(n_clients)]

    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        w = rng.dirichlet(alpha * np.ones(n_clients))

        pos = 0
        while pos < len(idx_c) and room.sum() > 0:
            p = w * (room > 0)
            p = p / p.sum() if p.sum() else (room > 0) / (room > 0).sum()
            take = np.minimum(rng.multinomial(len(idx_c) - pos, p), room)
            if take.sum() == 0:
                take = np.zeros(n_clients, int)
                take[rng.choice(n_clients, p=p)] = 1
            for i in np.flatnonzero(take):
                client_idx[i].extend(idx_c[pos:pos + take[i]])
                pos += take[i]
                room[i] -= take[i]

    return [np.array(ci) for ci in client_idx]

def get_alphas(K, config):
    return config['alpha_init'] / (1.0 + (np.arange(K)*config['alpha_decay']))

def parse_results(data):
    assert data['train']['num_nodes'] == data['sys']['num_nodes']
    assert data['train']['b'] == data['sys']['b']
    #assert data['train']['K'] == data['sys']['K']
    
    row = data.copy()
    row['n'] = data['sys']['num_nodes']
    row['sys_K'] = data['sys']['K']
    row['train_K'] = data['train']['K']
    row['b'] = data['sys']['b']
    row['atk_type'] = row['sys']['atk_type']
    row['train_atks'] = set(row['train']['train_atks'])
    row['test_atks'] = set(row['train']['test_atks'])
    
    del row['sys']
    del row['train']

    return row

def fetch_dataset(dataset):
    global_dataset = dict()
    if(dataset == 'MNIST'):
        mnist = fetch_openml('mnist_784',parser = 'auto')
        X = mnist.data
        y = mnist.target.astype(int).to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=GLOBAL_SEED)

        global_preproc = make_pipeline(
            MinMaxScaler(),
            PCA(n_components=0.93, svd_solver="full", random_state=GLOBAL_SEED, whiten=True),
            StandardScaler(),
        )

        X_pca = global_preproc.fit_transform(X_train)
        X_aug = np.hstack([X_pca, np.ones((X_pca.shape[0], 1))])
        X_pca_test = global_preproc.transform(X_test)
        X_aug_test = np.hstack([X_pca_test, np.ones((X_pca_test.shape[0], 1))])

        global_dataset = {
            'num_classes' : np.unique(y).shape[0],
            'X_train': X_aug,
            'y_train': y_train,
            'X_test': X_aug_test,
            'y_test': y_test
        }
    return global_dataset
