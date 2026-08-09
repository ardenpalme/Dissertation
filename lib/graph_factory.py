import networkx as nx
import numpy as np

from metrics import calc_nu_sq, effective_mixing
from utils import rng

class GraphFactory():
    def __init__(self, num_nodes, b):
        self.num_nodes = num_nodes
        self.b = b

    @staticmethod
    def calc_MH_weights(G):
        n = G.number_of_nodes()
        W = np.zeros((n, n))
        for i in range(n):
            for j in G.adj[i]:
                if(i != j):
                    W[i,j] = 1/(1+max(G.degree[i], G.degree[j]))
        np.fill_diagonal(W, np.ones(n) - (W @ np.ones(n)))
        return W

    @staticmethod
    def calc_generalized_MH_weights(G, target_pi):
        n = G.number_of_nodes()
        W = np.zeros((n, n))
        assert len(target_pi) == n
        for i in range(n):
            d_i = G.degree(i)
            if(d_i == 0):
                raise ValueError(f"node {i} has degree {int(G.degree(i))}")
            for j in G.adj[i]:
                d_j = G.degree(j)
                ratio = (target_pi[j] / d_j) / (target_pi[i] / d_i)
                W[i, j] = (1.0 / d_i) * min(1.0, ratio)
        np.fill_diagonal(W, 1.0 - W.sum(axis=1))
        return W

    @staticmethod
    def sample_until(gen, H, max_attempts=1000):
        for t in range(max_attempts):
            g = gen(t)
            if nx.is_connected(g.subgraph(H)):
                return g
        return None

    def create_graph(self, gtype, weights, seed, **kwargs):
        ''' 
        Watts-Strogatz:  If p is too low, graph is a ring
        Barabási-Albert: If m is too low, graph might have articulation points
        Geometric: If radius too small, graph is disconnected
        Erdos-Renyi: p = min(1.0, 3.0 * np.log(n) / n)
        Barbell: m1 = (n-2) // 2
        '''
        loc_rng = rng(seed)
        B = loc_rng.choice(np.arange(self.num_nodes), size=self.b, replace=False)
        H = np.array(list(set(np.arange(self.num_nodes)) - set(B)))
        G, W = None, None

        match gtype:
            case "random-regular":
                G = self.sample_until(lambda t: nx.random_regular_graph(kwargs['rand_reg_deg'], 
                                                                        self.num_nodes, seed=seed + t), H)
                  
            case "erdos-renyi":
                G = self.sample_until(lambda t: nx.gnp_random_graph(self.num_nodes, kwargs['er_p'], seed=seed + t), H)
                
            case "complete":
                G = nx.complete_graph(self.num_nodes)

            case "watts-strogatz":
                assert kwargs['ws_k']>=2 and (kwargs['ws_k']%2==0)
                assert 0.01 <= kwargs['ws_p'] and kwargs['ws_p'] <= 0.3
                G = self.sample_until(lambda t: nx.watts_strogatz_graph(
                    self.num_nodes, 
                    kwargs['ws_k'], 
                    kwargs['ws_p'],
                    seed=seed + t), H)

            case "barabasi-albert":
                G = self.sample_until(lambda t: nx.barabasi_albert_graph(
                    self.num_nodes, 
                    kwargs['ba_m'], 
                    seed=seed + t), H)

            case "geometric":
                G = self.sample_until(lambda t: nx.random_geometric_graph(
                    self.num_nodes, 
                    kwargs['geom_radius'], 
                    seed=seed + t), H)

        if G == None:
            raise ValueError(f"Could not generate a graph with connected honest subgraph")

        match weights:
            case "MH":
                W = self.calc_MH_weights(G)
            case "MH_gen":
                W = self.calc_generalized_MH_weights(G, kwargs['MH_target_pi'])
            
        return G, W, B, H

def calc_graph_metrics(W, b, H, gamma_C=0.4): # worst-case FPR
    
    L = -W.copy()
    np.fill_diagonal(L, 0.0)
    np.fill_diagonal(L, -L.sum(1))

    mu2 = float(np.sort(np.linalg.eigvalsh(L))[1])

    _, pi, lam_pi = effective_mixing(W, H, gamma_C)
    nu2, _ = calc_nu_sq(W, H, pi, gamma_C)
    s2 = lam_pi ** 2 + nu2
    g = 1.0 - s2
    
    return {'spectral_gap' : mu2, 'g':g}

def add_graph_plot(G, B, axis):
    pos = nx.kamada_kawai_layout(G)
    colors = ['red' if n in set(B) else 'lightblue' for n in G.nodes()]
    nx.draw(G, pos, node_color=colors, ax=axis, node_size=100)



