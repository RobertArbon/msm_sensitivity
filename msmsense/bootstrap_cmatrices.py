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

import pandas as pd
import numpy as np
import pyemma as pm
from pyemma.coordinates.data._base.datasource import DataSource
import mdtraj as md

from . import searchspace as cons
from .featurizers import distances, dihedrals
#
# def create_name(hp: Mapping) -> str:
#     feature_keys = [x for x in hp.keys() if x.startswith('feature')]
#     fname_list = []
#     for key in feature_keys:
#         elements = key.split('__')
#         if 'value' == elements[1] or hp['feature__value'] == elements[1]:
#             fname_list.append(f"{hp[key]}")
#     name = f"{'_'.join(fname_list)}.h5"
#     return name
#
# def filter_unique_hps(df: pd.DataFrame) -> pd.DataFrame:
#     unique_ixs = []
#     unique_names = []
#     for i, row in df.iterrows():
#         name = create_name(row.to_dict(into=dict))
#         if not name in unique_names:
#             unique_names.append(name)
#             unique_ixs.append(i)
#     return df.iloc[unique_ixs, :]

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


def estimate_cmatrices(trajs: List[np.ndarray]) -> Outputs:
    cmats = []
    lags = []
    for lag in cons.LAGS:
        m = pm.msm.estimate_markov_model(trajs, lag=lag, reversible=True, connectivity='largest',
                                             mincount_connectivity="1/n")
        cmat = m.count_matrix_active
        # logging.info(f"Estimated count matrix of size {cmat.shape} at lag {lag}")
        cmats.append(cmat)
        lags.append(lag)
    return Outputs(count_matrices=cmats, lags=np.array(lags))


def get_sub_dict(hp_dict: Dict[str, List[Union[str, int]]], name: str) -> Mapping:
    sub_dict = {k.split('__')[1]: v for k, v in hp_dict.items() if k.startswith(name)}
    return sub_dict


def discretize_trajectories(hp_dict: Dict[str, List[Union[str, int]]], trajs: List[np.ndarray]) -> List[np.ndarray]:
    tica = pm.coordinates.tica(trajs, **get_sub_dict(hp_dict, 'tica'))
    # # logging.info(f"Estimated tica")
    # logging.info(tica)
    y = tica.get_output()
    kmeans = pm.coordinates.cluster_kmeans(y, **get_sub_dict(hp_dict, 'cluster'), fixed_seed=2934798)
    # logging.info(f"Estimated kmeans")
    # logging.info(kmeans)
    z = kmeans.dtrajs
    z = [x.flatten() for x in z]
    return z


def get_probabilities(trajs: List[np.ndarray]) -> np.ndarray:
    lengths = np.array([x.shape[0] for x in trajs])
    probs = lengths/np.sum(lengths)
    return probs


def sample_trajectories(trajs: List[np.ndarray]) -> List[np.ndarray]:
    # ix = np.arange(len(trajs))
    # probs = get_probabilities(trajs)
    # sample_ix = np.random.choice(ix, size=ix.shape[0], p=probs, replace=True)
    # sampled_trajs = [trajs[i] for i in sample_ix]
    # return sampled_trajs
    return trajs


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


def do_bootstrap(hp_dict: Dict[str, List[Union[str, int]]], feat_trajs: List[np.ndarray]):
    # logging.info('in bootstrap')
    feat_trajs = sample_trajectories(feat_trajs)
    disc_trajs = discretize_trajectories(hp_dict, feat_trajs)
    outputs = estimate_cmatrices(disc_trajs)
    outputs.hp = hp_dict
    return outputs


def get_feature_trajs(traj_top_paths: Dict[str, List[Path]],hp_dict: Dict[str, List[Union[str, int]]]) -> List[np.ndarray]:
    trajs = get_trajs(traj_top_paths)
    feat_trajs = create_features(hp_dict, trajs)
    logging.info(f"Added features")
    return feat_trajs


def bootstrap_count_matrices(config: Tuple[str, Dict[str, List[Union[str, int]]]],
                             traj_top_paths: Dict[str, List[Path]],
                             output_dir: Path) -> None:
    """ Bootstraps the count matrices at a series of lag times.
    """
    hp_idx, hp_dict = config

    bs_dir = output_dir.joinpath(f"hp_{str(hp_idx)}")
    bs_dir.mkdir(exist_ok=True)

    ftrajs = get_feature_trajs(traj_top_paths, hp_dict)

    n_workers = min(cpu_count(), cons.BS_SAMPLES)
    pool = Pool(n_workers)
    logging.info(f"Bootstrapping hyper-parameter index value {hp_idx}")
    logging.info(f'Launching {cons.BS_SAMPLES} jobs on {n_workers} cores')

    results = []
    for i in range(cons.BS_SAMPLES):
        write_output = partial(write_matrices, sample_ix=i, out_dir=bs_dir)
        results.append(pool.apply_async(func=do_bootstrap, args=(hp_dict, ftrajs), callback=write_output))

    for r in results:
        r.get()

    pool.close()
    pool.join()
    logging.info(f'Finished boostrap hp_ix: {hp_idx}')


def get_input_trajs_top() -> Dict[str, List[Path]]:
    glob_str = cons.INPUT_TRAJ_GLOB
    trajs = list(Path('/').glob(f"{glob_str}/*.xtc"))
    top = list(Path('/').glob(f"{glob_str}/*.pdb"))[0]
    trajs.sort()
    assert trajs, 'no trajectories found'
    assert top, 'no topology found'
    return {'top': top, 'trajs': trajs}


def create_ouput_directory() -> Path:
    path = Path(cons.NAME)
    path.mkdir(exist_ok=True)
    return path


def get_hyperparameters(path: str) -> pd.DataFrame:
    hps = pd.read_hdf(path)
    logging.info(f"Hyper-parameter samples read from {str(path)}")
    logging.info(f"Hyper-parameter samples shape: {hps.shape}")
    logging.info(f"{hps.head()}")
    return hps


def setup_logger(out_dir: Path) -> None:
    logging.basicConfig(filename=str(out_dir.joinpath(f"{cons.NAME}.log")),
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main(hp_path: str) -> None:
    output_dir = create_ouput_directory()
    setup_logger(output_dir)
    hps = get_hyperparameters(hp_path)
    traj_top_paths = get_input_trajs_top()

    for i, row in hps.iterrows():
        # Making an explicit dict and str variable so that type hinting is explicit.
        hp = {k: v for k, v in row.to_dict().items()}
        ix = str(i)
        bootstrap_count_matrices((ix, hp), traj_top_paths, output_dir)


if __name__ == '__main__':
    main('./hp_sample.h5')

pm.msm.its