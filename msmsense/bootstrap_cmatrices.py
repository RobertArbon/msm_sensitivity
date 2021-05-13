"""
for a system (e.g., protein) saves features.
"""
from typing import Dict, List, Mapping, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import pickle
import logging
from multiprocessing import cpu_count, Pool
from functools import partial
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import pyemma as pm
from pyemma.coordinates.data._base.datasource import DataSource
import mdtraj as md

from .featurizers import distances, dihedrals


@dataclass
class Outputs:
    count_matrices: List[np.ndarray]
    lags: np.ndarray
    sample_ix: Optional[int] = None
    hp: Optional[Dict[str, Union[str, int]]] = None


def write_matrices(outputs: Outputs, out_dir: Path,
                   sample_ix: int) -> None:
    outputs.sample_ix = sample_ix
    file = out_dir.joinpath(f"{outputs.sample_ix}.pkl")
    d_outputs = {'count_matrices': outputs.count_matrices,
                 'lags': outputs.lags,
                 'sample_ix': outputs.sample_ix,
                 'hp': outputs.hp}
    pickle.dump(obj=d_outputs, file=file.open('wb'))


def estimate_cmatrices(trajs: List[np.ndarray], lags: List[int]) -> Outputs:
    cmats = []
    for lag in lags:
        cmat = None
        try:
            m = pm.msm.estimate_markov_model(trajs, lag=lag, reversible=True, connectivity='largest',
                                                 mincount_connectivity="1/n")
            cmat = m.count_matrix_active
        except RuntimeError:
            pass

        cmats.append(cmat)
    return Outputs(count_matrices=cmats, lags=np.array(lags))


def get_sub_dict(hp_dict: Dict[str, List[Union[str, int]]], name: str) -> Mapping:
    sub_dict = {k.split('__')[1]: v for k, v in hp_dict.items() if k.startswith(name)}
    return sub_dict


def discretize_trajectories(hp_dict: Dict[str, List[Union[str, int]]], trajs: List[np.ndarray],
                            seed: Union[int, None]) -> List[np.ndarray]:
    tica = pm.coordinates.tica(trajs, **get_sub_dict(hp_dict, 'tica'))
    y = tica.get_output()

    # np.save(Path('1FME/hp_0').joinpath('ttraj0.npy'), y[0])
    # tica.save('1FME/hp_0/tica.pm')
    kmeans = pm.coordinates.cluster_kmeans(y, **get_sub_dict(hp_dict, 'cluster'), fixed_seed=seed)
    z = kmeans.dtrajs
    # kmeans.save('1FME/hp_0/kmeans.pm')
    # np.save(Path('1FME/hp_0').joinpath('dtraj0.npy'), z[0])

    return z


def get_probabilities(trajs: List[np.ndarray]) -> np.ndarray:
    lengths = np.array([x.shape[0] for x in trajs])
    probs = lengths/np.sum(lengths)
    return probs


def sample_trajectories(trajs: List[np.ndarray], seed: Union[int, None]) -> List[np.ndarray]:
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    ix = np.arange(len(trajs))
    probs = get_probabilities(trajs)

    sample_ix = rng.choice(ix, size=ix.shape[0], p=probs, replace=True)
    sampled_trajs = [trajs[i] for i in sample_ix]
    return sampled_trajs


def create_features(hp_dict: Dict[str, List[Union[str, int]]], trajs: List[md.Trajectory]) -> List[np.ndarray]:
    feature = hp_dict['feature__value']
    if feature == 'dihedrals':
        feat_trajs = dihedrals(trajs=trajs, **get_sub_dict(hp_dict, 'dihedrals'))
    elif feature == 'distances':
        feat_trajs = distances(trajs=trajs, **get_sub_dict(hp_dict, 'distances'))
    else:
        raise RuntimeError('Feature not recognized')
    return feat_trajs


def get_trajs(traj_top_paths: Dict[str, List[Path]]) -> List[md.Trajectory]:
    traj_paths = [str(x) for x in traj_top_paths['trajs']]
    top = str(traj_top_paths['top'])
    trajs = [md.load(x, top=top) for x in traj_paths]
    return trajs


def do_bootstrap(hp_dict: Dict[str, List[Union[str, int]]], feat_trajs: List[np.ndarray], seed: Union[int, None],
                 lags: List[int]):
    # logging.info('in bootstrap')
    feat_trajs = sample_trajectories(feat_trajs, seed)
    disc_trajs = discretize_trajectories(hp_dict, feat_trajs, seed)
    outputs = estimate_cmatrices(disc_trajs, lags)
    outputs.hp = hp_dict
    return outputs


def get_feature_trajs(traj_top_paths: Dict[str, List[Path]],hp_dict: Dict[str, List[Union[str, int]]]) -> List[np.ndarray]:
    trajs = get_trajs(traj_top_paths)
    feat_trajs = create_features(hp_dict, trajs)
    logging.info(f"Added features")
    return feat_trajs


def bootstrap_count_matrices(config: Tuple[str, Dict[str, List[Union[str, int]]]],
                             traj_top_paths: Dict[str, List[Path]], seed: int,
                             bs_samples: int, lags: List[int], output_dir: Path) -> None:
    """ Bootstraps the count matrices at a series of lag times.
    """
    hp_idx, hp_dict = config

    bs_dir = output_dir.joinpath(f"hp_{str(hp_idx)}")
    bs_dir.mkdir(exist_ok=True)

    ftrajs = get_feature_trajs(traj_top_paths, hp_dict)
    np.save(bs_dir.joinpath('ftraj0.npy'), ftrajs[0])

    n_workers = min(cpu_count(), bs_samples)
    pool = Pool(n_workers)
    logging.info(f"Bootstrapping hyper-parameter index value {hp_idx}")
    logging.info(f'Launching {bs_samples} jobs on {n_workers} cores')

    results = []
    for i in range(bs_samples):
        write_output = partial(write_matrices, sample_ix=i, out_dir=bs_dir)
        results.append(pool.apply_async(func=do_bootstrap, args=(hp_dict, ftrajs, seed, lags), callback=write_output))

    for r in results:
        r.get()

    pool.close()
    pool.join()
    logging.info(f'Finished boostrap hp_ix: {hp_idx}')


def get_input_trajs_top(data_dir: Path, top_path: Path, traj_glob: str) -> Dict[str, List[Path]]:
    trajs = list(data_dir.glob(traj_glob))
    top_path = data_dir.joinpath(top_path)
    if len(trajs) == 0:
        raise RuntimeError('No trajectories found')
    if not top_path.exists():
        raise RuntimeError(f"Topolgy path doesn't exist at:\n{top_path}")

    top = str(top_path)
    trajs.sort()
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
    logging.info(f"{hps.head()}")
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
        bootstrap_count_matrices((ix, hp), traj_top_paths, args.seed, args.num_repeats, lags, output_dir)
        raise Exception('Fin')

def configure_parser(sub_subparser: ArgumentParser):
    p = sub_subparser.add_parser('count_matrices')
    p.add_argument('-i', '--hp-sample', type=Path, help='Path to file that contains the hyperparameter samples')
    p.add_argument('-d', '--data-dir', type=Path, help='Base directory used to determine trajectory and topology paths')
    p.add_argument('-t', '--topology-path', type=Path, help='Topology path')
    p.add_argument('-g', '--trajectory-glob', type=str, help='Trajectory glob string relative to --data-dir')
    p.add_argument('-r', '--num-repeats', type=int, help='Number of bootstrap samples')
    p.add_argument('-l', '--lags', type=str, help='Lags as a Python range specification start:end:stride',
                   default='2:51:2')
    p.add_argument('-o', '--output-dir', type=Path, help='Path to output directory')
    p.add_argument('-s', '--seed', type=int, help='Random seed', default=None)
    p.set_defaults(func=main)
# if __name__ == '__main__':
#     main('./hp_sample.h5')
#
# pm.msm.its