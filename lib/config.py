GLOBAL_SEED = 12345
NUM_NODES = 20
BASE_CONF = {
    'alpha_init':0.5,
    'alpha_decay': 0.1,
    'graph_type':'random-regular',
    'graph_weights':'MH_gen',
    'graph_args': {},
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
        # 'gamma_C': 0.12, this is now a vector
        'beta_C': 0.1,
        'clf_model':'rbf',
        'train_atks': ['label_flip', 'sign_flip', 'gaussian', 'ALIE', 'IPM'],
        'test_atks': ['label_flip', 'sign_flip', 'gaussian', 'ALIE', 'IPM'],
    }
}

# These classifier data generation and system simulation parameters are identical
BASE_CONF['train']['num_nodes'] = BASE_CONF['sys']['num_nodes']
BASE_CONF['train']['b'] = BASE_CONF['sys']['b']
BASE_CONF['train']['K'] = BASE_CONF['sys']['K']
