import networkx as nx
import numpy as np

class GraphFactory():
    def __init__(self, num_nodes, b, seed):
        self.num_nodes = num_nodes
        self.b = b
        self.seed = seed 

    @staticmethod
    def calc_unif_weights(G):
        n = G.number_of_nodes()
        W = np.zeros((n, n))
        for i in range(n):
            nbrs = [j for j in G.adj[i] if j != i]
            w = 1 / (1 + len(nbrs))
            for j in nbrs:
                W[i, j] = w
            W[i, i] = w
        return W

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
    def calc_srw_weights(G):
        n = G.number_of_nodes()
        W = np.zeros((n, n))
        for i in range(n):
            deg = G.degree(i)
            if deg == 0:
                W[i, i] = 1.0
            else:
                for j in G.adj[i]:
                    W[i, j] = 1.0 / deg
        return W

    def calc_lazy_srw_weights(self, G):
        n = G.number_of_nodes()
        W = self.calc_srw_weights(G)
        return 0.5 * np.eye(n) + 0.5 * W

    @staticmethod
    def calc_pagerank_weights(G, alpha=0.85): #TODO should be kwarg
        n = G.number_of_nodes()
        A = nx.adjacency_matrix(G).todense()
        D_inv = np.diag(1 / np.array([G.degree(i) for i in range(n)], dtype=float))
        P_rw = D_inv @ A
        v = np.ones((n, 1)) / n  # Uniform teleportation
        P_pr = alpha * P_rw + (1 - alpha) * (np.ones((n, 1)) @ v.T)
        return P_pr

    @staticmethod
    def calc_random_row_stochastic(G, seed=42): #TODO should be related to global seed
        n = G.number_of_nodes()
        np.random.seed(seed)
        W = np.random.rand(n, n) + 0.1
        for i in range(n):
            for j in range(n):
                if i == j or j not in G.adj[i]:
                    W[i, j] = 0.0
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 
        W = W / row_sums
        return W

    @staticmethod
    def sample_until(gen, b, max_attempts=1000):
        for t in range(max_attempts):
            g = gen(t)
            if nx.node_connectivity(g) >= b + 1:
                return g
        return None

    def create_graph(self, gtype, weights, **kwargs):
        ''' 
        Watts-Strogatz:  If p is too low, graph is a ring
        Barabási-Albert: If m is too low, graph might have articulation points
        Geometric: If radius too small, graph is disconnected
        Erdos-Renyi: p = min(1.0, 3.0 * np.log(n) / n)
        Barbell: m1 = (n-2) // 2
        '''
        G, W = None, None
        subset = np.arange(self.num_nodes)
        match gtype:
            case "random-regular":
                G = self.sample_until(lambda t: nx.random_regular_graph(kwargs['rand_reg_deg'], self.num_nodes, seed=self.seed + t), self.b)
                  
            case "erdos-renyi":
                # p = min(1.0, 3.0 * np.log(self.num_nodes) / self.num_nodes)
                G = self.sample_until(lambda t: nx.gnp_random_graph(self.num_nodes, kwargs['er_p'], seed=self.seed + t), self.b)
                
            case "complete":
                G = nx.complete_graph(self.num_nodes)

            case "watts-strogatz":
                assert kwargs['ws_k']>=2 and (kwargs['ws_k']%2==0)
                assert 0.01 <= kwargs['ws_p'] and kwargs['ws_p'] <= 0.3
                G = self.sample_until(lambda t: nx.watts_strogatz_graph(
                    self.num_nodes, 
                    kwargs['ws_k'], 
                    kwargs['ws_p'],
                    seed=self.seed + t), self.b)

            case "barabasi-albert":
                G = self.sample_until(lambda t: nx.barabasi_albert_graph(
                    self.num_nodes, 
                    kwargs['ba_m'], # number of edges to attach from a new node to existing nodes
                    seed=self.seed + t), self.b)

            case "barbell":
                #assert self.b==1, "connectivity is min(m1, m2) = 2, must have b<=1"
                G = nx.barbell_graph(kwargs['barbell_m1'], kwargs['barbell_m2'])
                bridge_nodes = [i for i in range(self.num_nodes) if G.degree(i) == 2]
                neighbors_of_bridge = set()
                for i in bridge_nodes:
                    neighbors_of_bridge.update(G.neighbors(i))
                subset = np.array(list(set(range(self.num_nodes)) - (set(bridge_nodes) | neighbors_of_bridge)))

            case "geometric":
                G = self.sample_until(lambda t: nx.random_geometric_graph(
                    self.num_nodes, 
                    kwargs['geom_radius'], 
                    seed=self.seed + t), self.b)

            case "torus_2d":
                assert self.b<=3, "2D torus has vertex-connectivity 4"
                side = int(np.sqrt(self.num_nodes))
                if side**2 != self.num_nodes:
                    raise ValueError("self.num_nodes must be a perfect square for 2D torus.")
                G = nx.grid_2d_graph(side, side, periodic=True)
                G = nx.convert_node_labels_to_integers(G)

            case "hypercube":
                k = int(np.log2(self.num_nodes))
                if 2**k != self.num_nodes:
                    raise ValueError("self.num_nodes must be a power of 2 for hypercube.")
                if k <= self.b:
                    raise ValueError(f"Hypercube dimension {k} is not > b={self.b}. Connectivity too low.")
                G = nx.hypercube_graph(k)

        if G == None:
            print(f"Could not generate a {gtype} graph with κ(G)>{self.b}")
            return None

        match weights:
            case "MH":
                W = self.calc_MH_weights(G)
            case "MH_gen":
                W = self.calc_generalized_MH_weights(G, kwargs['MH_target_pi'])
            case "unif":
                W = self.calc_unif_weights(G)
            case "SRW":
                W = self.calc_srw_weights(G)
            case "lazy_SRW":
                W = self.calc_lazy_srw_weights(G)
            case "pagerank":
                W = self.calc_pagerank_weights(G)
            case "rand_row_stoch":
                W = self.calc_random_row_stochastic(G)
            
        return G, W, subset

def calc_graph_metrics(G,W,b):
    n = W.shape[0]
    
    L = -W.copy()
    np.fill_diagonal(L, 0.0)
    np.fill_diagonal(L, -L.sum(1))

    mu2 = float(np.sort(np.linalg.eigvalsh(L))[1])
    D_b = float(np.sort(W - np.diag(np.diag(W)), axis=1)[:, -b:].sum(1).max())
    spectral_gap = float(np.abs(np.linalg.eigvalsh(W - np.ones((n, n)) / n)).max())
    
    return {'mu2' : mu2, 'D_b' : D_b, 'spectral_gap': spectral_gap}

def add_graph_plot(G, B, axis):
    pos = nx.kamada_kawai_layout(G) # nx.spring_layout(G)
    colors = ['red' if n in set(B) else 'lightblue' for n in G.nodes()]
    nx.draw(G, pos, node_color=colors, with_labels=True, ax=axis)



