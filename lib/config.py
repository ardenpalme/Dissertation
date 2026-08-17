import numpy as np

# Global Variables
GLOBAL_SEED = 12345
NUM_NODES = 20
GRAPHS = ['random-regular',
          'watts-strogatz',
          'geometric',
          'erdos-renyi',
          'barabasi-albert']
PERMUTED_GRAPHS = np.roll(np.asarray(GRAPHS), -1).tolist()
GRAPH_ABBREV = {
    'random-regular': 'RR',
    'watts-strogatz': 'WS',
    'geometric': 'RGG',
    'erdos-renyi': 'ER',
    'barabasi-albert': 'BA',
}
GRAPH_FULL_NAMES = {
    'random-regular': 'Random Regular',
    'watts-strogatz': 'Watts-Strogatz',
    'geometric': 'Random Geometric',
    'erdos-renyi': 'Erdos-Renyi',
    'barabasi-albert': 'Barabasi-Albert',
}

ATKS = ['label_flip', 'sign_flip', 'gaussian', 'ALIE', 'IPM']
ATK_ABBREV = {
    'label_flip': 'Label Flip',
    'sign_flip': 'Sign Flip',
    'gaussian': 'Gaussian',
    'ALIE': 'ALIE',
    'IPM': 'IPM',
}

ALGS = ['RDSGD', 'ORACLE_0', 'ORACLE_1', 'IOS', 'SCC', 'TriMean', 'CooMed']

BASE_CONF = {
    'reg_param': 1,
    'alpha_init': 0.5,
    'alpha_decay': 0.25,
    'graph_type': 'random-regular',
    'graph_weights': 'MH_gen',
    'graph_args': {
        'ba_m': 5,
        'ws_k': 6,
        'ws_p': 0.1,
        'rand_reg_deg': 10,
        'geom_radius': 0.55,
        'er_p': min(1.0, 3.0 * np.log(NUM_NODES) / NUM_NODES),
    },
    'sys': {
        'num_nodes': NUM_NODES,
        'b': 5,
        'K': 200,
        'atk_type': 'sign_flip',
        'fpr_spread': 0.003,
        'threat_model': 'T3'
    },
    'train': {
        'fpr_mean': 0.2,
        'fpr_spread': 0.05,
        'gamma_C': np.clip((0.05*np.linspace(-1, 1, NUM_NODES)) + 0.2, 0, 0.95),
        'beta_C': 0.1,
        'clf_model': 'xgb',
        'train_atks': ['label_flip', 'sign_flip', 'gaussian', 'ALIE', 'IPM'],
        'test_atks': ['label_flip', 'sign_flip', 'gaussian', 'ALIE', 'IPM'],
    },
    'batch_sz': 100,
    'data_heterogeneity': 100,
    'pi': (np.full(NUM_NODES, 1.0/NUM_NODES)),
    'oracle_params': [
        # Perfect Oracle
        {'C_fnr': np.full(NUM_NODES, 0), 'C_fpr': np.full(NUM_NODES, 0)},
        # Threat Model IV Oracle
        {'C_fnr': np.full(NUM_NODES, 1), 'C_fpr': np.full(NUM_NODES, 0)},
    ]
}

# These classifier data generation and system simulation parameters identical
BASE_CONF['train']['num_nodes'] = BASE_CONF['sys']['num_nodes']
BASE_CONF['train']['b'] = BASE_CONF['sys']['b']
BASE_CONF['train']['K'] = BASE_CONF['sys']['K']
