import os, sys, copy, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from fastnanoid import generate
from datetime import datetime
from joblib import Parallel, delayed

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
lib_path = project_root / "lib"
sys.path.insert(0, str(lib_path))
sys.path.insert(0, str(project_root))

from lib.utils import rng 
from lib.simulation import clf_stage, sim_task
from lib.config import BASE_CONF, NUM_NODES, GRAPHS, PERMUTED_GRAPHS, ALGS, ATKS, GRAPH_ABBREV

def run_cycle(config, exp_name, permute_graphs=False):
    RUN_DIR = os.path.join(Path().resolve(), exp_name)
    IMAGES_DIR = os.path.join(RUN_DIR, "images")
    COLORS = dict(zip(ALGS, ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']))
    SEEDS = [111, 333,777,888,222]
    CACHE = {}
    b = config['train']['b']

    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    exp_name_str = " " + exp_name + " "; print(exp_name_str.center(len("=" * 56), '='))

    # Save experiment configuration
    payload = copy.deepcopy(config)
    del payload['train']['gamma_C']
    del payload['pi']
    del payload['oracle_params']
    with open(os.path.join(RUN_DIR, f'config.json'), 'w') as f:
        def convert(o):
            if isinstance(o, np.generic):
                return o.item()
            raise TypeError(f'{type(o)} not serializable')
        json.dump(payload, f, indent=2, default=convert)

    for g_idx,g in enumerate(GRAPHS):
        cfg = copy.deepcopy(config)
        cfg['graph_type'] = g
        if permute_graphs:
            cfg['sys']['graph_type'] = PERMUTED_GRAPHS[g_idx]
        CACHE[g] = cfg

    # Train Byzantine classifier in parallel over 5 topologies
    res1 = Parallel(n_jobs=5, backend='loky', inner_max_num_threads=6, verbose=10)(
        delayed(clf_stage)(g, CACHE[g], RUN_DIR, b) for g in GRAPHS)
    P1 = dict(res1)
    print("Byzantine classifier trained")

    # Run system simulation
    jobs = [(g, atk, s) for g in GRAPHS for atk in ATKS for s in SEEDS]
    res2 = Parallel(n_jobs=30, backend='loky', inner_max_num_threads=1, verbose=10)(
        delayed(sim_task)(g, CACHE[g], atk, s, P1[g]['proj_const'],
                          P1[g]['preproc'], P1[g]['est'], P1[g]['gamma_sys'],
                          dict(C_fpr=P1[g]['clf_thr']['fpr_ach'],
                               C_fnr=P1[g]['clf_thr']['fnr_ach'],
                               C_tau=P1[g]['clf_thr']['C_tau']),
                          ALGS, b, SEEDS)
        for g, atk, s in jobs)

    R = {(g, atk, s): (df, consts) for g, atk, s, df, consts in res2}
    rows, met_rows, clf_rows = [], [], []
    fig1, ax1 = plt.subplots(len(GRAPHS), len(ATKS),
                             figsize=(3 * len(ATKS), 3 * len(GRAPHS)),
                             squeeze=False, sharex=True)
    all_parts = []
    for g_idx, g in enumerate(GRAPHS):
        ax = ax1[g_idx]
        clf_rows.append({'graph': g, 'proj_const': P1[g]['proj_const'],
                         **P1[g]['op_pt'], **P1[g]['metrics']})
        for atk_idx, atk in enumerate(ATKS):
            for s in SEEDS:
                df, consts = R[(g, atk, s)]
                first = (s == SEEDS[0])
                if first:
                    df_reset = df.reset_index()
                    df_reset['graph'] = g
                    df_reset['attack'] = atk
                    all_parts.append(df_reset)
                    met_rows.append({**consts, 'atk': atk, 'graph': g})
                for alg in ALGS:
                    col = 'test_acc_pi' if alg == 'RDSGD' else 'test_acc'
                    rows.append({'alg': alg, 'atk': atk, 'seed': s,
                                 'graph': g, 'test_acc': df.loc[alg, col].iloc[-1]})
                    if first:
                        V = df.loc[alg, 'C_unif'] + np.pow(df.loc[alg, 'opt_gap'], 2)
                        if(atk_idx == 0): 
                            if(g_idx == len(GRAPHS)-1):
                                ax[atk_idx].plot(V, color=COLORS[alg], label=alg)
                                ax[atk_idx].set_xlabel='Iteration (k)'
                            else:
                                ax[atk_idx].plot(V, color=COLORS[alg])
                            ax[atk_idx].set_ylabel("Lyapunov Function $V^{k}$")
                        else: 
                            ax[atk_idx].plot(V, color=COLORS[alg])
                            ax[atk_idx].set_ylabel("")
                        if(permute_graphs):
                            ax[atk_idx].set(title=f'({GRAPH_ABBREV[g]}, {GRAPH_ABBREV[PERMUTED_GRAPHS[g_idx]]}, {atk})', yscale='log')
                        else:
                            ax[atk_idx].set(title=f'({g}, {atk})', yscale='log')

    df_complete = pd.DataFrame()
    df_complete = pd.concat(all_parts, ignore_index=True).set_index(['graph', 'attack', 'alg', 'k'])
    df_complete.to_csv(os.path.join(RUN_DIR, f'sys_sim.csv'))
                       
    pd.DataFrame(clf_rows).set_index('graph').to_csv(os.path.join(RUN_DIR, f'clf_metrics.csv'))
    pd.DataFrame(met_rows).set_index(['graph','atk']).to_csv(os.path.join(RUN_DIR, f'sys_metrics.csv'))

    df_all = pd.DataFrame(rows)
    df_summary = df_all.groupby(['graph','atk','alg'])['test_acc'].agg(['mean','std']).unstack('alg')
    df_summary.to_csv(os.path.join(RUN_DIR, "test_accuracy.csv"))
                        
    fig1.legend(loc="upper right")
    fig1.tight_layout()
    fig1.savefig(os.path.join(IMAGES_DIR, f'conv_plots.png'))

if __name__ == '__main__':
    # 1. Threat Model I Baseline
    config = copy.deepcopy(BASE_CONF)
    config['sys']['b'] = 0
    config['train']['b'] = 0
    config['sys']['threat_model'] = 'T0'
    run_cycle(config, 'tm_0_baseline')

    # 2. Threat Model III Baseline
    config = copy.deepcopy(BASE_CONF)
    run_cycle(config, 'tm_3_baseline')

    # 3. High Byzantine Influence
    config = copy.deepcopy(BASE_CONF)
    config['train']['b'] = 8
    config['sys']['b'] = config['train']['b']
    config['graph_args']['geom_radius'] = 0.6
    run_cycle(config, 'high_byz')

    # 4. High Data Heterogeneity
    config = copy.deepcopy(BASE_CONF)
    config['data_heterogeneity'] = 2
    run_cycle(config, 'high_data_het')

    # 5. Stress Test
    config = copy.deepcopy(BASE_CONF)
    config['train']['b'] = 8
    config['sys']['b'] = config['train']['b']
    config['graph_args']['geom_radius'] = 0.6
    config['data_heterogeneity'] = 2
    run_cycle(config, 'stress_test')

    # 6. Skewed Edge Weights
    config = copy.deepcopy(BASE_CONF)
    config['pi-dir-alpha'] = 2
    pi = rng('dir','graphs').dirichlet(np.full(NUM_NODES,config['pi-dir-alpha']))
    config['pi'] = (pi / pi.sum())
    run_cycle(config, 'skewed_edge_weights')

    # 7. Worst Case
    config = copy.deepcopy(BASE_CONF)
    config['train']['b'] = 8
    config['sys']['b'] = config['train']['b']
    config['graph_args']['geom_radius'] = 0.6
    config['pi-dir-alpha'] = 2
    pi = rng('dir','graphs').dirichlet(np.full(NUM_NODES,config['pi-dir-alpha']))
    config['pi'] = (pi / pi.sum())
    config['data_heterogeneity'] = 2
    run_cycle(config, 'worst_case')

    # 8. Graph Permutation
    config = copy.deepcopy(BASE_CONF)
    run_cycle(config, 'permute_graphs', permute_graphs=True)

