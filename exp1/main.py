import sys
import copy
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import pandas as pd
import seaborn as sns
from pathlib import Path
from fastnanoid import generate
from datetime import datetime

project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))
lib_path = project_root / "lib"
sys.path.insert(0, str(lib_path))        
sys.path.insert(0, str(project_root))

from lib.graph_factory import GraphFactory, add_graph_plot
from lib.proj_const_estimator import ProjConstEstimator
from lib.utils import fetch_dataset, get_alphas, rng, seed, save_to_pkl, load_from_pkl
from lib.classifier import ByzClassifier, load_run, save_run
from lib.system import SystemSimulator
from lib.metrics import MetricsCalculator
from lib.config import BASE_CONF, NUM_NODES

# =========== Helper Functions =============
fmt = lambda sec : f"{sec // 60}m {sec % 60:.2f}s"

def convert(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f'{type(o)} not serializable')

def get_graph_args(num_byz_nodes, tar_dir_het_param):
    target_pi = rng('dir','graphs').dirichlet(np.full(NUM_NODES, tar_dir_het_param))
    grap_args = {
        'ws_k': 6,
        'ws_p': 0.1,
        'rand_reg_deg':10,
        'geom_radius':0.23,
        'er_p':min(1.0, 3.0 * np.log(NUM_NODES) / NUM_NODES),
        'MH_target_pi':target_pi,
    }
    if(num_byz_nodes == 5):
        r_ub = {
            'erdos-renyi': 0.1461,
            'random-regular': 0.1195,
            'watts-strogatz': 0.0711,
            'geometric': 0.0625,
            'complete': 0.3438
        }
        return grap_args, r_ub
        
    elif(num_byz_nodes == 9):
        grap_args['geometric'] = 0.4
        
        r_ub = {
            'erdos-renyi': 0.043,
            'random-regular': 0.0,
            'watts-strogatz': 0.0,
            'geometric': 0.0,
            'complete': 0.0799
        }
        return grap_args, r_ub
    else: raise ValueError(f'No graph parameters for b={num_byz_nodes}')

# Experiment Configuration
RUN_DIR = os.path.join(Path().resolve(), "default")
IMAGES_DIR = os.path.join(RUN_DIR, "images")
config = copy.deepcopy(BASE_CONF)
config['train']['b'] = 5
config['sys']['b'] = config['train']['b'] 
config['train']['clf_model'] = 'xgb'
config['data_heterogeneity'] = 100
config['reg_param'] = 1
config['tar_dir_het_param'] = 1
b = config['train']['b']
graph_args, r_ub = get_graph_args(b, config['tar_dir_het_param'])
config['graph_args'] = graph_args
config['graph_weights'] = 'MH'

# Notebook Variables
graphs = ['random-regular','watts-strogatz', 'geometric', 'erdos-renyi']
ATKS = ['label_flip', 'sign_flip', 'gaussian', 'ALIE', 'IPM']
ALGS = ['RDSGD', 'ORACLE', 'IOS', 'SCC', 'TriMean', 'CooMed']
ALGS_TEST = ['RDSGD', 'ORACLE']
COLORS = dict(zip(ALGS, ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']))
SEEDS = [111,222,333,555,777]
DEBUG = True
plots = [plt.subplots(1,len(ATKS),figsize=(15,3), sharey=True) for g in graphs]
ta_tables = dict()

if __name__ == '__main__':
    # Save experiment configuration to disk
    payload = copy.deepcopy(config)
    del payload['graph_args']['MH_target_pi']
    with open(os.path.join(RUN_DIR, f'config.json'), 'w') as f:
        json.dump(payload, f, indent=2, default=convert)

    global_dataset = fetch_dataset('MNIST')
    gf = GraphFactory(config['train']['num_nodes'], b)
    proj_const_estimator = ProjConstEstimator(config, global_dataset, gf) 
    clf = ByzClassifier(config, global_dataset, gf)
    sys_sim = SystemSimulator(config, global_dataset, gf)
    mc = MetricsCalculator(config, global_dataset, rng('metrics-calculator'))

    for g_idx, g in enumerate(graphs):
        config['graph_type']=g
        config['rand_dropout'] = min(r_ub[g], 0.2)
        GRAPH_DIR = os.path.join(RUN_DIR, g)
        os.makedirs(GRAPH_DIR, exist_ok=True)
        fig, ax = plots[g_idx]
        if DEBUG: 
            print(f" {g} graph ".center(56, '='))
        else:
            print(f"{g} graph simulation", end='')
            
        t0 = time.perf_counter()
        
        # Generate simulation data for classifier
        pkl_file = os.path.join(RUN_DIR,f'clf_sim_data_{g}_{config['sys']['b']}.pkl.gz')
        if os.path.exists(pkl_file):
            if DEBUG: print("clf simulations loaded from disk")
            conf = load_run(clf, pkl_file)
            proj_const = conf['proj_const']
        else:
            if DEBUG: print("clf simulations", end='')
            t1 = time.perf_counter()
            proj_const_estimator.configure(config, seed('proj-const-estimator'))
            proj_const = proj_const_estimator.estimate()
            clf_sim_data = clf.run_simulations(config, proj_const, seed('byz-clf','sim'), is_printing=False)
            save_run(clf, pkl_file, {'proj_const':proj_const})
            if DEBUG: print(f" took {fmt(time.perf_counter() - t1)}")
        
        clf_est, clf_preproc = clf.fit()
        clf_op_pt = clf.calc_optimal_op_pt()
        clf_metrics = clf.test()
        
        prelim_metrics = dict()
        prelim_metrics['proj_const'] = proj_const
        prelim_metrics.update(clf_op_pt)
        prelim_metrics.update(clf_metrics)
        if DEBUG: print(f"clf performance: val (γ={clf_op_pt['C_fpr']:.3f}, β={clf_op_pt['C_fnr']:.3f}) test (γ={clf_metrics['gamma_C']:.3f}, β={clf_metrics['beta_C']:.3f})")

        # Record Preliminary Metrics per graph
        clf_metrics_file_path = os.path.join(RUN_DIR, f'clf_metrics_{config['sys']['b']}.csv')
        clf_metrics = copy.deepcopy(prelim_metrics)
        clf_metrics.update({'graph':g})
        if not os.path.exists(clf_metrics_file_path):
            df_clf_metrics = pd.DataFrame([clf_metrics]).set_index('graph')
        else:
            df_clf_metrics = pd.read_csv(clf_metrics_file_path).set_index('graph')
            
            new_row = pd.DataFrame([clf_metrics]).set_index('graph')
            if g in df_clf_metrics.index:
                df_clf_metrics.loc[g] = new_row.loc[g]
            else:
                df_clf_metrics = pd.concat([df_clf_metrics, new_row])
        df_clf_metrics.to_csv(clf_metrics_file_path)

        # Run system simulation
        rows = []
        for atk_idx, atk in enumerate(ATKS):
            for s_idx, s in enumerate(SEEDS): 
                sim_params = {
                    'algorithms': ALGS,      
                    'atk_type': atk,
                    'threat_model': 'T3',
                    'seed': s
                }
                
                params_C = dict()
                params_C['C_fpr'] = clf_metrics['gamma_C']
                params_C['C_fnr'] = clf_metrics['beta_C']
                params_C['C_tau'] = clf_op_pt['C_tau']

                if DEBUG: print(f"simulation ({atk} attack, seed {s})", end='')
                t1 = time.perf_counter()
                sys_sim.init_simulation(config, proj_const, clf_preproc, clf_est, params_C, seed('sys', s), is_printing_logs=False)
                df = sys_sim.simulate(sim_params, dropout=config['rand_dropout'])
                if(s_idx == 0): df.to_csv(os.path.join(GRAPH_DIR,f'sys_sim_df_{g}_{b}_{atk}.csv'))
                if DEBUG:
                    if(s_idx == 0): print(f" took {fmt(time.perf_counter() - t1)}", end=' ')
                    else: print(f" took {fmt(time.perf_counter() - t1)}")
                    

                # Records for final summary df 
                for alg_idx, alg in enumerate(ALGS): 
                    if(alg == 'RDSGD'):
                        # ASSERTION ONLY TRUE FOR UNIFORM MH WEIGHTS!!
                        # assert(df.loc[alg, 'test_acc_pi'].iloc[-1] == df.loc[alg, 'test_acc'].iloc[-1]) 
                        test_acc = df.loc[alg, 'test_acc_pi'].iloc[-1]
                    else:
                        test_acc = df.loc[alg, 'test_acc'].iloc[-1]

                    # Plot Lyapunov function for that graph for that attack
                    if(s_idx == 0):
                        gap = np.pow(df.loc[alg,'opt_gap'],2)
                        conv = df.loc[alg,'C_unif']
                        ax[atk_idx].plot(conv+gap, color=COLORS[alg])
                        ax[atk_idx].set(title=f"({g}, {atk})", yscale='log')
                        if(atk_idx == 0): ax[atk_idx].set_ylabel("Lyapunov $V^{k}$")
                        else: ax[atk_idx].set_ylabel("")
                        if DEBUG:
                            if(alg_idx == 0): print("(")
                            if(alg_idx != len(ALGS)-1): print(f"{alg}:{test_acc:.3f}",end=', ') 
                            else: print(f"{alg}:{test_acc:.3f})")
                            
                    rows.append({
                        'alg':alg,
                        'atk':atk,
                        'seed':s,
                        'test_acc': test_acc,
                    })
                    
                # Record system simulation metrics per (graph, atk) tuple
                if(s_idx == 0):
                    metrics_file_path = os.path.join(RUN_DIR, f'sys_metrics_{config['sys']['b']}.csv')
                    rdsgd_consts = mc(sys_sim.get_sim_config(), sys_sim.models, clf_metrics['gamma_C'], clf_metrics['beta_C'], proj_const)
                    pi, x_opt, x_pi_opt = rdsgd_consts['pi'], rdsgd_consts['x_opt'], rdsgd_consts['x_pi_opt']
                    del rdsgd_consts['pi']
                    del rdsgd_consts['x_opt']
                    del rdsgd_consts['x_pi_opt']
                    metrics_payload = copy.deepcopy(rdsgd_consts)
                    metrics_payload.update({'atk':atk, 'graph':g})

                    # Update local .csv file
                    if not os.path.exists(metrics_file_path):
                        df_metrics = pd.DataFrame([metrics_payload])
                        df_metrics.set_index(['graph','atk'],inplace=True)
                    else:
                        df_metrics = pd.read_csv(metrics_file_path)
                        df_metrics.set_index(['graph','atk'],inplace=True)
                        
                        new_row = pd.DataFrame([metrics_payload]).set_index(['graph', 'atk'])
                        if (g, atk) in df_metrics.index:
                            df_metrics.loc[(g, atk)] = new_row.loc[(g, atk)]
                        else:
                            df_metrics = pd.concat([df_metrics, new_row])
                    df_metrics.to_csv(metrics_file_path)

        if not DEBUG: print(f" took {fmt(time.perf_counter() - t0)}")

        # save attack-level convergence for one seed for this graph topology
        os.makedirs(IMAGES_DIR, exist_ok=True)
        fig.savefig(os.path.join(IMAGES_DIR, f'conv_plots_{g}_{config['sys']['b']}.png'))
        
        # Record test accuracy final df for that graph 
        summary_file_path = os.path.join(GRAPH_DIR, f'test_acc_table.csv')
        df_summary = pd.DataFrame(rows).groupby(['atk', 'alg'])['test_acc'].agg(['mean', 'std']).unstack('alg')
        df_summary.to_csv(summary_file_path)
        if DEBUG:
            print(f"{g} graph test accuracy table")
            print(df_summary)


