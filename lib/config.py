import numpy as np

GLOBAL_SEED = 12345
NUM_NODES = 20
TARGET_PI_DIRICHLET_ALPHA = 10
LOGFILE = "results.csv"
CONFIG_RNG = np.random.default_rng(GLOBAL_SEED)

uniform_dist = (1/NUM_NODES) * np.ones(NUM_NODES)

BASE_CONF = {
    'alpha_init':0.5,
    'alpha_decay': 0.1,
    'graph_type':'random-regular',
    'graph_weights':'MH_gen',
    'graph_args': {
        #'geom_radius': 0.45,
        'er_p':min(1.0, 3.0 * np.log(NUM_NODES) / NUM_NODES),
        'rand_reg_deg':5,
        'tar_pi_dir_alpha': TARGET_PI_DIRICHLET_ALPHA,
        'MH_target_pi':CONFIG_RNG.dirichlet(np.full(NUM_NODES, TARGET_PI_DIRICHLET_ALPHA))
    },
    'reg_param':0.2,
    'batch_sz':100, 
    'data_heterogeneity': 1,
    'sys' : {
        'num_nodes': NUM_NODES,
        'b': 5,
        'K': 200,
        'atk_type':'sign_flip',
    },
    'train': {
        'gamma_C': 0.12,
        'beta_C': 0.1,
        'clf_model':'rbf',
        'train_atks': ['label_flip', 'sign_flip', 'gaussian', 'ALIE', 'IPM'], # same attacks for val
        'test_atks': ['label_flip', 'sign_flip', 'gaussian', 'ALIE', 'IPM'],
    }
}
BASE_CONF['train']['num_nodes'] = BASE_CONF['sys']['num_nodes']
BASE_CONF['train']['b'] = BASE_CONF['sys']['b']
BASE_CONF['train']['K'] = BASE_CONF['sys']['K']
