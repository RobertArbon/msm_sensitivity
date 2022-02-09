"""
for a system (e.g., protein) saves features.
"""
from typing import Dict, List, Mapping, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
import pickle
import logging
from multiprocessing import cpu_count, Pool
from functools import partial
from argparse import ArgumentParser

from msmtools.estimation import transition_matrix as _transition_matrix
from msmtools.analysis import timescales as _timescales
from pyemma.util.metrics import vamp_score

import pandas as pd
import numpy as np
import pyemma as pm
from pyemma.coordinates.data._base.datasource import DataSource
import mdtraj as md

from .featurizers import distances, dihedrals


MAX_PROCS = 20  # don't want more than this many processes


@dataclass
class Outputs:
    vamp_by_lag_by_proc: Dict[int, Dict[int, np.ndarray]]
    ts_by_lag_by_proc: Dict[int, Dict[int, np.ndarray]]
    ix: int = None


def write_outputs(outputs: Outputs, out_path: Path) -> None:
    d_outputs = {'ts': outputs.ts_by_lag_by_proc,
                 'vamp': outputs.vamp_by_lag_by_proc,
                 'hp_ix': outputs.ix}
    pickle.dump(obj=d_outputs, file=out_path.open('wb'))


def vamp(cmat, T, method, k):
    C0t = cmat
    C00 = np.diag(C0t.sum(axis=1))
    Ctt = np.diag(C0t.sum(axis=0))
    return vamp_score(T, C00, C0t, Ctt, C00, C0t, Ctt,
                      k=k, score=method)


def score_cmats(cmats_by_lag) -> Outputs:
    ts_by_lag_by_proc = dict()
    vamp_by_lag_by_proc = dict()
    for lag, cmat in cmats_by_lag.items():
        if cmat is not None:
            # Transition matrix
            t_mat = _transition_matrix(cmat, reversible=True)
            # timescales
            ts = _timescales(t_mat, tau=lag)
            ts = ts[1:]
            num_its = min(MAX_PROCS, int(np.sum(ts > lag)))
            proc_labels = (np.arange(num_its)+2).astype(int)
            ts_by_lag_by_proc[int(lag)] = dict(zip(proc_labels, ts[:num_its]))
            # VAMP scores
            vamp_by_lag_by_proc[int(lag)] = {k: vamp(cmat, t_mat, method='VAMP2', k=k) for k in proc_labels}

    outputs = Outputs(vamp_by_lag_by_proc=vamp_by_lag_by_proc,
                      ts_by_lag_by_proc=ts_by_lag_by_proc)
    return outputs


def estimate_cmatrices(trajs: List[np.ndarray], lags: List[int]) -> Dict[int, np.ndarray]:
    cmats_by_lag = dict()
    for lag in lags:
        cmat = None
        try:
            m = pm.msm.estimate_markov_model(trajs, lag=lag, reversible=True, connectivity='largest',
                                                 mincount_connectivity="1/n")
            cmat = m.count_matrix_active
        except RuntimeError:
            pass

        cmats_by_lag[int(lag)] = cmat
    return cmats_by_lag


def get_sub_dict(hp_dict: Dict[str, List[Union[str, int]]], name: str) -> Mapping:
    sub_dict = {k.split('__')[1]: v for k, v in hp_dict.items() if k.startswith(name)}
    return sub_dict


def discretize_trajectories(hp_dict: Dict[str, List[Union[str, int]]], trajs: List[np.ndarray],
                            seed: Union[int, None]) -> List[np.ndarray]:
    tica = pm.coordinates.tica(trajs, **get_sub_dict(hp_dict, 'tica'))
    y = tica.get_output()

    kmeans = pm.coordinates.cluster_kmeans(y, **get_sub_dict(hp_dict, 'cluster'), fixed_seed=seed)
    z = kmeans.dtrajs
    return z


def get_probabilities(trajs: List[np.ndarray]) -> np.ndarray:
    lengths = np.array([x.shape[0] for x in trajs])
    probs = lengths/np.sum(lengths)
    return probs


def get_rng(seed: Union[int, None]) -> Any:
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    return rng


def sample_trajectories(trajs: List[np.ndarray], rng: Any, randomize: bool = True) -> Tuple[List[np.ndarray], np.ndarray]:
    ix = np.arange(len(trajs))
    if randomize:
        probs = get_probabilities(trajs)
        sample_ix = rng.choice(ix, size=ix.shape[0], p=probs, replace=True)
    else:
        sample_ix = ix
    sampled_trajs = [trajs[i] for i in sample_ix]
    return sampled_trajs, sample_ix


def create_features(hp_dict: Dict[str, List[Union[str, int]]], trajs_top_dict: Dict[str, List[str]]) -> List[np.ndarray]:
    feature = hp_dict['feature__value']
    logging.info(f"Creating {feature} feature")
    if feature == 'dihedrals':
        feat_trajs = dihedrals(traj_top_dict=trajs_top_dict, **get_sub_dict(hp_dict, 'dihedrals'))
    elif feature == 'distances':
        feat_trajs = distances(traj_top_dict=trajs_top_dict, **get_sub_dict(hp_dict, 'distances'))
    else:
        raise RuntimeError('Feature not recognized')
    return feat_trajs


def get_trajs(traj_top_paths: Dict[str, List[Path]]) -> List[md.Trajectory]:
    traj_paths = [str(x) for x in traj_top_paths['trajs']]
    top = str(traj_top_paths['top'])
    trajs = [md.load(x, top=top) for x in traj_paths]
    return trajs


def do_bootstrap(hp_dict: Dict[str, List[Union[str, int]]], feat_trajs: List[np.ndarray], seed: Union[int, None],
                 lags: List[int], out_dir: Path, hp_idx: int):
    disc_trajs = discretize_trajectories(hp_dict, feat_trajs, seed)
    cmats_by_lag = estimate_cmatrices(disc_trajs, lags)
    outputs = score_cmats(cmats_by_lag)
    outputs.ix = hp_idx
    write_outputs(outputs, out_dir)
    return True


def get_feature_trajs(traj_top_paths: Dict[str, List[str]],hp_dict: Dict[str, List[Union[str, int]]]) -> List[np.ndarray]:
    traj_top_paths['trajs'].sort()
    feat_trajs = create_features(hp_dict, traj_top_paths)
    logging.info(f"Added features")
    return feat_trajs


def bootstrap_count_matrices(config: Tuple[str, Dict[str, List[Union[str, int]]]],
                             traj_top_paths: Dict[str, List[str]], seed: int,
                             bs_samples: int, n_cores: int, lags: List[int], output_dir: Path) -> None:
    """ Bootstraps the count matrices at a series of lag times.
    """
    hp_idx, hp_dict = config
    hp_idx = int(hp_idx)
    
    bs_dir = output_dir.joinpath(f"hp_{str(hp_idx)}")
    bs_dir.mkdir(exist_ok=True)

    logging.info(f"Getting feature trajectories")
    all_ftrajs = get_feature_trajs(traj_top_paths, hp_dict)
    rng = get_rng(seed)

    n_workers = min(n_cores, bs_samples)
    logging.info(f"Bootstrapping hyper-parameter index value {hp_idx}")
    results = []
    bs_dict = dict()
    if n_workers > 1:
        pool = Pool(n_workers)
        logging.info(f'Launching {bs_samples} jobs on {n_workers} cores')
        for i in range(bs_samples):
            ftrajs, _ = sample_trajectories(all_ftrajs, rng, bs_samples > 1)
            results.append(pool.apply_async(func=do_bootstrap,
                                            args=(hp_dict, ftrajs, seed, lags,
                                                  bs_dir.joinpath(f"{i}.pkl"), hp_idx)))
    
        for r in results:
            r.get()
    
        pool.close()
        pool.join()
    else: 
        for i in range(bs_samples):
            ftrajs, _ = sample_trajectories(all_ftrajs, rng, bs_samples > 1)
            do_bootstrap(hp_dict, ftrajs, seed, lags, bs_dir.joinpath(f"{i}.pkl"), hp_idx)
    logging.info(f'Finished boostrap hp_ix: {hp_idx}')


def get_input_trajs_top(data_dir: Path, top_path: Path, traj_glob: str) -> Dict[str, List[str]]:
    trajs = list(data_dir.glob(traj_glob))
    top_path = data_dir.joinpath(top_path)
    if len(trajs) == 0:
        raise RuntimeError('No trajectories found')
    if not top_path.exists():
        raise RuntimeError(f"Topolgy path doesn't exist at:\n{top_path}")

    top = str(top_path)
    trajs.sort()
    trajs = [str(x) for x in trajs]
    logging.info(f"Topology loaded: {top}")
    logging.info(f"{len(trajs)} trajectories loaded: \n{trajs[:5]}\n...\n{trajs[-5:]}")
    return {'top': top, 'trajs': trajs}


def create_ouput_directory(path: Path) -> Path:
    path.mkdir(exist_ok=True)
    return path


def get_hyperparameters(path: str) -> pd.DataFrame:
    hps = pd.read_hdf(path)
    logging.info(f"Hyper-parameter samples read from {str(path)}")
    logging.info(f"Hyper-parameter samples shape: {hps.shape}")
    logging.info(f"\n{hps.head()}")
    return hps


def setup_logger(out_dir: Path) -> None:
    logging.basicConfig(filename=str(out_dir.joinpath('log')),
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def parse_lags(rng: str) -> List[int]:
    x = rng.split(":")
    y = [int(i) for i in x]
    lags = list(range(*y))
    return lags


def main(args, parser) -> None:
    output_dir = create_ouput_directory(args.output_dir.absolute())
    setup_logger(output_dir)
    hps = get_hyperparameters(args.hp_sample)
    traj_top_paths = get_input_trajs_top(args.data_dir.absolute(), args.topology_path, args.trajectory_glob)
    lags = parse_lags(args.lags)
    for i, row in hps.iterrows():
        # Making an explicit dict and str variable so that type hinting is explicit.
        hp = {k: v for k, v in row.to_dict().items()}
        ix = str(i)
        logging.info(f"Running hyperparameters: {row}")
        bootstrap_count_matrices((ix, hp), traj_top_paths, args.seed, args.num_repeats, args.num_cores, lags, output_dir)


def configure_parser(sub_subparser: ArgumentParser):
    p = sub_subparser.add_parser('count_matrices')
    p.add_argument('-i', '--hp-sample', type=Path, help='Path to file that contains the hyperparameter samples')
    p.add_argument('-d', '--data-dir', type=Path, help='Base directory used to determine trajectory and topology paths')
    p.add_argument('-t', '--topology-path', type=Path, help='Topology path')
    p.add_argument('-g', '--trajectory-glob', type=str, help='Trajectory glob string relative to --data-dir')
    p.add_argument('-r', '--num-repeats', type=int, help='Number of bootstrap samples')
    p.add_argument('-n', '--num-cores', type=int, help='Number of cpu cores to use.', default=1)
    p.add_argument('-l', '--lags', type=str, help='Lags as a Python range specification start:end:stride',
                   default='2:51:2')
    p.add_argument('-o', '--output-dir', type=Path, help='Path to output directory')
    p.add_argument('-s', '--seed', type=int, help='Random seed', default=None)
    p.set_defaults(func=main)