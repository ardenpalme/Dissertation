import os, copy, warnings
from pathlib import Path
import numpy as np

from graph_factory import GraphFactory
from proj_const_estimator import ProjConstEstimator
from classifier import ByzClassifier, load_run, save_run
from system import SystemSimulator
from metrics import MetricsCalculator
from utils import fetch_dataset, rng, seed

DATA_CACHE = os.path.join(Path(__file__).resolve().parent, 'mnist_pca.npz')

warnings.simplefilter("ignore", UserWarning)

def load_dataset(path=DATA_CACHE):
    if os.path.exists(path):
        z = np.load(path)
        return {'num_classes': int(z['num_classes']),
                'X_train': z['X_train'], 'y_train': z['y_train'],
                'X_test':  z['X_test'],  'y_test':  z['y_test']}
    gd = fetch_dataset('MNIST')
    np.savez(path, num_classes=gd['num_classes'],
             X_train=gd['X_train'], y_train=gd['y_train'],
             X_test=gd['X_test'],   y_test=gd['y_test'])
    return gd

def clf_stage(g, config_g, run_dir, b):
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
    gd = load_dataset() 
    gf = GraphFactory(config_g['train']['num_nodes'], b)
    clf = ByzClassifier(config_g, gd, gf)

    pkl_file = os.path.join(run_dir, f"clf_sim_data_{g}_{b}.pkl.gz")
    if os.path.exists(pkl_file):
        proj_const = load_run(clf, pkl_file)['proj_const']
    else:
        config_g['graph_args']['MH_target_pi'] = (1 - config_g['train']['gamma_C']) * config_g['pi']
        
        pce = ProjConstEstimator(config_g, gd, gf)
        pce.configure(config_g, seed('proj-const-estimator'))
        proj_const = pce.estimate()
        
        clf.run_simulations(config_g, proj_const, seed('byz-clf','sim'), is_printing=False)
        save_run(clf, pkl_file, {'proj_const': proj_const})

    clf_est, clf_preproc = clf.fit()
    clf_op_pt = clf.calc_optimal_op_pt()
    gamma_sys = np.clip(clf_op_pt['C_fpr'] + (np.linspace(-1, 1, config_g['sys']['num_nodes']) * config_g['sys']['fpr_spread']), 0, 0.95)
    clf_thr = clf.threshold_at_fpr(gamma_sys)
    
    return g, dict(proj_const=proj_const, est=clf_est, preproc=clf_preproc,
                   op_pt=clf_op_pt, clf_thr=clf_thr, metrics=clf.test(), gamma_sys=gamma_sys)

def sim_task(g, config_g, atk, s, proj_const, preproc, est, gamma_sys, rdsgd_params, algs, b, seeds): 
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
    gd = load_dataset() 
    gf = GraphFactory(config_g['train']['num_nodes'], b)
    ss = SystemSimulator(config_g, gd, gf)

    stationary_dist = (1 - gamma_sys) * config_g['pi']
    config_g['graph_args']['MH_target_pi'] = stationary_dist
    #config_g['oracle_params'].append({'C_fpr':rdsgd_params['C_fpr'], 'C_fnr':rdsgd_params['C_fnr']})
    #algs.append('ORACLE_2')
    ss.init_simulation(config_g, proj_const, preproc, est, config_g['oracle_params'], rdsgd_params, seed('sys', s), is_printing_logs=False)
    df = ss.simulate(dict(algorithms=algs, atk_type=atk, threat_model='T3', seed=s))
    
    consts = None
    if s == seeds[0]:
        mc = MetricsCalculator(config_g, gd, rng('metrics-calculator'))
        consts = mc(ss.get_sim_config(), ss.models, rdsgd_params['C_fpr'], rdsgd_params['C_fnr'], proj_const, stationary_dist)
        for k in ('pi', 'x_opt', 'x_pi_opt'): consts.pop(k, None)
    return g, atk, s, df, consts
