import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from fastnanoid import generate
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
lib_path = project_root / "lib"
sys.path.insert(0, str(lib_path))        
sys.path.insert(0, str(project_root))   

from lib.graph_factory import GraphFactory, add_graph_plot
from lib.proj_const_estimator import ProjConstEstimator
from lib.utils import fetch_dataset, rng, seed
from lib.classifier import ByzClassifier
from lib.system import SystemSimulator
from lib.config import BASE_CONF, GLOBAL_SEED, LOGFILE

if __name__ == '__main__':
    config = BASE_CONF
    global_dataset = fetch_dataset('MNIST')
    gf = GraphFactory(config['train']['num_nodes'], config['train']['b'], seed('graph'))
    proj_const_estimator = ProjConstEstimator(config, global_dataset, rng('proj-const-estimator')) 
    classifier = ByzClassifier(config, global_dataset, rng('byz-classifier'))
    sys_sim = SystemSimulator(config, global_dataset, rng('ml-system-sim'), LOGFILE)

    proj_const_estimator.configure(gf,config)
    proj_const = proj_const_estimator.estimate()

    RUN_DIR = Path(__file__).parent
    sim_params = {
    'algorithms':['RDSGD_ORACLE'],
    'atk_type':config['sys']['atk_type'],
    'threat_model':'T3'
    }
    num_test_pts = 6
    xv, yv = np.meshgrid(np.linspace(0,1,num_test_pts)[:-1], np.linspace(0,1,num_test_pts), indexing='ij')
    for i in range(num_test_pts-1): # cannot allow full FPR
        for j in range(num_test_pts): 
            run_results = dict()
            RUN_ID = generate()
            run_results['timestamp'] = pd.to_datetime(datetime.now())
            run_results['proj_const'] = proj_const

            params_C = dict(C_fpr=xv[i,j], C_fnr=yv[i,j], C_tau=0) 
            run_results.update(params_C)

            sys_sim.configure(gf, config, proj_const, None, None, params_C, is_printing_logs=False)
            df= sys_sim.simulate(sim_params, run_results)
            run_results['max_test_acc'] = df.loc['RDSGD_ORACLE']['max_test_acc'].max()
            run_results['min_test_acc'] = df.loc['RDSGD_ORACLE']['min_test_acc'].min()
            run_results['term_cons'] = df.loc['RDSGD_ORACLE']['Ck'].iloc[-1]
            print(f"T3_RDSGD_ORACLE(fpr={params_C['C_fpr']:.3f}, fnr={params_C['C_fnr']:.3f})= \
                  [{run_results['min_test_acc']},{run_results['max_test_acc']}]")
            payload = sys_sim.log_results(run_results, RUN_DIR, RUN_ID)

