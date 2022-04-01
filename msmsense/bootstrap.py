"""
for a system (e.g., protein) saves features.
"""
from typing import Dict, List, Mapping, Tuple, Optional, Union, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
import pickle
import logging
from multiprocessing import cpu_count, Pool
from functools import partial
from argparse import ArgumentParser

from pyemma.coordinates.transform import TICA
from pyemma.coordinates.clustering import KmeansClustering
from pyemma.util.metrics import vamp_score
from scipy.stats import entropy

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


def score_msms(mods_by_lag: Dict[int, pm.msm.MaximumLikelihoodMSM]) -> Outputs:
    ts_by_lag_by_proc = dict()
    vamp_by_lag_by_proc = dict()
    for lag, mod in mods_by_lag.items():
        if mod is not None:
            # timescales
            ts = mod.timescales()
            num_its = min(MAX_PROCS, int(np.sum(ts > lag)))
            proc_labels = (np.arange(num_its)+2).astype(int)
            ts_by_lag_by_proc[int(lag)] = dict(zip(proc_labels, ts[:num_its]))
            # VAMP scores
            cmat = mod.count_matrix_active
            tmat = mod.transition_matrix
            vamp_by_lag_by_proc[int(lag)] = {k: vamp(cmat, tmat, method='VAMP2', k=k) for k in proc_labels}

    outputs = Outputs(vamp_by_lag_by_proc=vamp_by_lag_by_proc,
                      ts_by_lag_by_proc=ts_by_lag_by_proc)
    return outputs


def estimate_msms(trajs: List[np.ndarray], lags: List[int]) -> Dict[int, pm.msm.MaximumLikelihoodMSM]:
    mods_by_lag = dict()
    for lag in lags:
        cmat = None
        try:
            m = pm.msm.estimate_markov_model(trajs, lag=lag, reversible=True, connectivity='largest',
                                                 mincount_connectivity="1/n")
            mods_by_lag[int(lag)] = m
        except RuntimeError:
            print(f'Error with model lag {lag}')

    return mods_by_lag


def get_sub_dict(hp_dict: Dict[str, List[Union[str, int]]], name: str) -> Mapping:
    sub_dict = {k.split('__')[1]: v for k, v in hp_dict.items() if k.startswith(name)}
    return sub_dict


def discretize_trajectories(hp_dict: Dict[str, List[Union[str, int]]], trajs: List[np.ndarray],
                            seed: Union[int, None]) -> Tuple[TICA, KmeansClustering]:
    tica = pm.coordinates.tica(trajs, **get_sub_dict(hp_dict, 'tica'))
    y = tica.get_output()
    kmeans = pm.coordinates.cluster_kmeans(y, **get_sub_dict(hp_dict, 'cluster'), fixed_seed=seed)
    return tica, kmeans


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
    logging.debug('in sample trajectories')
    ix = np.arange(len(trajs))
    if randomize:
        logging.debug('randomizing')
        probs = get_probabilities(trajs)
        sample_ix = rng.choice(ix, size=ix.shape[0], p=probs, replace=True)
    else:
        sample_ix = ix
    logging.debug('resampling')
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


def bs_score(hp_dict: Dict[str, List[Union[str, int]]],
             feat_trajs: List[np.ndarray], bs_ix: np.ndarray, seed: Union[int, None],
             out_dir: Path, hp_idx: int,
             lags: List[int]):
    try:
        tica, kmeans = discretize_trajectories(hp_dict, feat_trajs, seed)
        disc_trajs = kmeans.dtrajs
        mods_by_lag = estimate_msms(disc_trajs, lags)
        outputs = score_msms(mods_by_lag)
        outputs.ix = hp_idx
        write_outputs(outputs, out_dir)
    except Exception as e:
        logging.info(e)
    return True


def get_feature_trajs(traj_top_paths: Dict[str, List[str]],hp_dict: Dict[str, List[Union[str, int]]]) -> List[np.ndarray]:
    traj_top_paths['trajs'].sort()
    feat_trajs = create_features(hp_dict, traj_top_paths)
    logging.info(f"Added features")
    return feat_trajs


def get_all_projections(msm: pm.msm.MaximumLikelihoodMSM, num_procs: int, dtrajs: List[np.ndarray]) -> List[np.ndarray]:
    """ Project dtrajs onto first num_proc eigenvectors excluding stationary distribution. i.e., if num_proc=1 then project onto the slowest eigenvector only. 
    All projections ignore the stationary distribution
    """
    evs = msm.eigenvectors_right(num_procs+1)
    active_set = msm.active_set
    NON_ACTIVE_PROJ_VAL = np.nan # if the state is not in the active set, set the projection to this value. 
    NON_ACTIVE_IX_VAL = -1
    evs = evs[:, 1:] # remove the stationary distribution
    proj_trajs = []
    for dtraj in dtrajs:
        all_procs = []
        for proc_num in range(num_procs):

            tmp = np.ones(dtraj.shape[0], dtype=float)
            tmp[:] = NON_ACTIVE_PROJ_VAL

            for i in range(dtraj.shape[0]):
                x = msm._full2active[dtraj[i]]
                if x != NON_ACTIVE_IX_VAL:
                    tmp[i] = evs[x, proc_num]
                tmp = tmp.reshape(-1, 1)

            all_procs.append(tmp)
        all_procs = np.concatenate(all_procs, axis=1)
        proj_trajs.append(all_procs)

    return proj_trajs


def mixing_ent(x):
    x = np.abs(x)
    return entropy(x)


def msm_projection_trajectories(hp_dict, feat_trajs, seed, lag, processes) -> List[np.array]: 
    # Estimate MSM
    tica, kmeans = discretize_trajectories(hp_dict, feat_trajs, seed)
    disc_trajs = kmeans.dtrajs

    mod = estimate_msms(disc_trajs, [lag])[lag]
    # Project onto MSM right eigenvectors
    ptrajs = get_all_projections(mod, processes, disc_trajs)
    return ptrajs


def msm_projection_dataframe(ptrajs: List[np.ndarray], bs_traj_paths: List[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples([(bs_traj_paths[i], j) for i in range(len(bs_traj_paths)) for j in range(ptrajs[i].shape[0])])
    ptrajs_all = np.concatenate(ptrajs, axis=0)
    ptrajs_df = pd.DataFrame(ptrajs_all, index=index, columns=[f"{i + 2}" for i in range(ptrajs[0].shape[1])])
    # Calculate the purity of the projections
    ptrajs_df['mixing'] = ptrajs_df.apply(mixing_ent, axis=1)
    ptrajs_df.dropna(inplace=True)
    return ptrajs_df


def sample_ev(n_ev: int, n_cut: int, ptrajs_df, top_path: str, threshold: float=1e-6) -> md.Trajectory:
    n_ev = str(n_ev)
    df = ptrajs_df.loc[:, [n_ev, 'mixing']].copy(deep=True)
    df['cat'] = pd.qcut(df[n_ev], q=n_cut,  duplicates='drop')
    df['min'] = df.groupby('cat')['mixing'].transform('min')
    df = df.loc[np.abs(df['mixing'] - df['min']) < threshold, :]
    sample = df.groupby('cat').sample(n=1)
    sample.sort_values(by='cat', inplace=True)
    sample_ixs = list(sample.index)
    traj = md.join([md.load_frame(x, top=top_path, index=y) for x, y in sample_ixs])
    return traj


def bs_ev_sample(hp_dict: Dict[str, List[Union[str, int]]],
             feat_trajs: List[np.ndarray], bs_ix: np.ndarray, seed: Union[int, None],
             out_path: Path, hp_idx: int,
             lag: int, processes: int, num_cuts: int, traj_paths: List[str], top_path: str) -> bool:
    bs_traj_paths = [traj_paths[i] for i in bs_ix]
    ptrajs = msm_projection_trajectories(hp_dict, feat_trajs, seed, lag, processes)
    ptrajs_df = msm_projection_dataframe(ptrajs, bs_traj_paths)

    # out_path assumes we want to save everything to a pickle, but we want to create a new directory
    out_dir = out_path.with_suffix("")
    out_dir.mkdir(exist_ok=True, parents=True)

    result = dict()
    for i in range(2, processes+2):
        traj = sample_ev(i, num_cuts, ptrajs_df, top_path)
        traj.save_xtc(filename=str(out_dir.joinpath(f'ev_{i}.xtc')))
    result['bs_ix'] = bs_ix
    result['hp_ix'] = hp_idx
    pickle.dump(obj=result, file=out_dir.joinpath('metadata.pkl').open('wb'))
    return True


def project_bs(config, traj_top_paths, seed, n_cores, output_dir, lag, project_dir, project_ixs):

    hp_idx, hp_dict = config
    hp_idx = int(hp_idx)

    logging.info(f"Getting feature trajectories")
    all_ftrajs = get_feature_trajs(traj_top_paths, hp_dict)

    for project_ix in project_ixs:
        # get bootstrap samples
        bs_main_dir = project_dir.joinpath(f"hp_{project_ix}")
        bs_dirs = list(bs_main_dir.glob('*'))
        bs_dirs.sort()
        bs_samples = len(bs_dirs)

        # output dir
        out_bs_dir = output_dir.joinpath(f"hp_{str(hp_idx)}_ev_{str(project_ix)}")
        out_bs_dir.mkdir(exist_ok=True)

        n_workers = min(n_cores, bs_samples)

        logging.info(f"Bootstrapping projections onto index value {hp_idx}")
        results = []
        if n_workers > 1:
            pool = Pool(n_workers)
            logging.info(f'Launching {bs_samples} jobs on {n_workers} cores')
            for bs_dir in bs_dirs:
                results.append(pool.apply_async(func=project_single_evs,
                                                args=(hp_dict,
                                                      lag,
                                                      traj_top_paths['top'],
                                                      all_ftrajs,
                                                      seed,
                                                      out_bs_dir.joinpath(bs_dir.name).with_suffix('.pkl'),
                                                      bs_dir,
                                                      )
                                                )
                               )

            for r in results:
                r.get()

            pool.close()
            pool.join()
        else:
            pass
            # for i in range(bs_samples):
            #     ftrajs, bs_ix = sample_trajectories(all_ftrajs, rng, bs_samples > 1)
            #     bs_func(hp_dict, ftrajs, bs_ix, seed, bs_dir.joinpath(f"{i}.pkl"), hp_idx, **kwargs)
        logging.info(f'Finished boostrap hp_ix: {hp_idx}')


def get_ttps_from_dir(dir: Path, top: str) -> Dict[str, Union[str, List[str]]]:
    ev_traj_paths = list(dir.glob('*.xtc'))
    ev_traj_paths.sort()
    ev_traj_paths = [str(x) for x in ev_traj_paths]
    return {'trajs': ev_traj_paths, 'top': top}


def randomize_trajs_from_metadata(dir: Path, trajs: List[np.ndarray]) -> List[np.ndarray]:
    ev = pickle.load(dir.joinpath('metadata.pkl').open('rb'))
    bs_ix = ev.pop('bs_ix')
    _ = ev.pop('hp_ix')
    return [trajs[i] for i in bs_ix]


def project_single_evs(hp_dict: Dict, lag: int, top_path: str, all_ftrajs: List[np.ndarray], seed: int,
                       output_path: Path, ev_dir: Path):

    ttps = get_ttps_from_dir(ev_dir, top_path)
    ev_ftrajs = get_feature_trajs(ttps, hp_dict)
    ftrajs = randomize_trajs_from_metadata(ev_dir, all_ftrajs)

    tica, kmeans = discretize_trajectories(hp_dict, ftrajs, seed)
    disc_trajs = kmeans.dtrajs
    mod = estimate_msms(disc_trajs, [lag])[lag]
    results = dict()
    for proc, ftraj in enumerate(ev_ftrajs):
        # discretize the sample
        ev_ttrajs = tica.transform([ftraj])
        ev_dtrajs = kmeans.transform(ev_ttrajs)
        ev_dtrajs = [x.flatten() for x in ev_dtrajs]
        # Project onto MSM right eigenvectors
        ptrajs = get_all_projections(mod, len(ev_ftrajs), ev_dtrajs)
        results[proc+2] = ptrajs

    pickle.dump(obj=results, file=output_path.open('wb'))


def bootstrap(config: Tuple[str, Dict[str, List[Union[str, int]]]],
              traj_top_paths: Dict[str, List[str]], seed: int,
              bs_samples: int, n_cores: int, output_dir: Path, bs_func: Callable[..., bool],
              **kwargs) -> None:
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
    if n_workers > 1:
        with Pool(n_workers) as pool:
            logging.info(f'Launching {bs_samples} jobs on {n_workers} cores')
            for i in range(bs_samples):
                logging.info('sampling trajectories')
                logging.info(f"size: {np.sum([x.shape[0]*x.shape[1] for x in all_ftrajs])*64/8/1024/1024/1024}")
                ftrajs, bs_ix = sample_trajectories(all_ftrajs, rng, bs_samples > 1)
                logging.info('appending ', i, ' to queue. ')
                results.append(pool.apply_async(func=bs_func,
                                                args=(hp_dict, ftrajs, bs_ix, seed,
                                                      bs_dir.joinpath(f"{i}.pkl"), hp_idx),
                                                kwds=kwargs))
    
            for r in results:
                r.get()
    
        # pool.close()
        # pool.join()
    else: 
        for i in range(bs_samples):
            logging.info('running ', i)
            ftrajs, bs_ix = sample_trajectories(all_ftrajs, rng, bs_samples > 1)

            logging.info('starting bs_func')
            bs_func(hp_dict, ftrajs, bs_ix, seed, bs_dir.joinpath(f"{i}.pkl"), hp_idx, **kwargs)
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


def get_hyperparameters(path: str, hp_ixs: List[int]) -> pd.DataFrame:
    hps = pd.DataFrame(pd.read_hdf(path))
    if len(hp_ixs):
        hps = hps.iloc[hp_ixs, :]
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


def score(hp_sample, hp_ixs, data_dir, topology_path, trajectory_glob, num_repeats, num_cores, lags, output_dir, seed) -> None:
    output_dir = create_ouput_directory(output_dir.absolute())
    setup_logger(output_dir)
    hps = get_hyperparameters(hp_sample, hp_ixs)
    print(data_dir, topology_path, trajectory_glob)
    traj_top_paths = get_input_trajs_top(data_dir.absolute(), topology_path, trajectory_glob)
    lags = parse_lags(lags)
    for i, row in hps.iterrows():
        # Making an explicit dict and str variable so that type hinting is explicit.
        hp = {k: v for k, v in row.to_dict().items()}
        ix = str(i)
        logging.info(f"Running hyperparameters: {row}")
        bootstrap(config=(ix, hp),
                  traj_top_paths=traj_top_paths,
                  seed=seed, bs_samples=num_repeats, n_cores=num_cores,
                  output_dir=output_dir,
                  bs_func=bs_score, lags=lags)


def sample_evs(lag, processes, num_cuts, hp_sample, data_dir, topology_path, trajectory_glob, num_repeats, num_cores,
             output_dir, seed, hp_ixs):
    output_dir = create_ouput_directory(output_dir.absolute())
    setup_logger(output_dir)
    print('HP_IXs', hp_ixs)
    hps = get_hyperparameters(hp_sample, hp_ixs)
    traj_top_paths = get_input_trajs_top(data_dir.absolute(), topology_path, trajectory_glob)
    for i, row in hps.iterrows():
        # Making an explicit dict and str variable so that type hinting is explicit.
        hp = {k: v for k, v in row.to_dict().items()}
        ix = str(i)
        logging.info(f"Running hyperparameters: {row}")
        bootstrap(config=(ix, hp),
                  traj_top_paths=traj_top_paths,
                  seed=seed, bs_samples=num_repeats, n_cores=num_cores,
                  output_dir=output_dir,
                  bs_func=bs_ev_sample, lag=lag, processes=processes, num_cuts=num_cuts,
                  traj_paths=traj_top_paths['trajs'], top_path=traj_top_paths['top'])


def project_evs(lag, hp_sample, data_dir, topology_path, trajectory_glob, num_cores,
                   output_dir, seed, hp_ix, project_dir, project_ixs):
    output_dir = create_ouput_directory(output_dir.absolute())
    setup_logger(output_dir)
    hps = get_hyperparameters(hp_sample, [hp_ix])
    traj_top_paths = get_input_trajs_top(data_dir.absolute(), topology_path, trajectory_glob)

    # Making an explicit dict and str variable so that type hinting is explicit.
    hp = {k: v[hp_ix] for k, v in hps.to_dict().items()}
    logging.info(f"Running hyperparameters: {hp}")
    project_bs(config=(hp_ix, hp),
              traj_top_paths=traj_top_paths,
              seed=seed, n_cores=num_cores,
              output_dir=output_dir,
              lag=lag,
              project_dir=project_dir, project_ixs=project_ixs)


def dump_dtrajs(hp_sample, data_dir, topology_path, trajectory_glob, output_dir, seed, hp_ixs) -> None:
    output_dir = create_ouput_directory(output_dir.absolute())
    setup_logger(output_dir)
    print('HP_IXs', hp_ixs)
    hps = get_hyperparameters(hp_sample, hp_ixs)
    traj_top_paths = get_input_trajs_top(data_dir.absolute(), topology_path, trajectory_glob)
    for hp_idx, row in hps.iterrows():
        hp = {k: v for k, v in row.to_dict().items()}
        logging.info(f"Running hyperparameters: {row}")

        hp_dir = output_dir.joinpath(f"hp_{str(hp_idx)}")
        hp_dir.mkdir(exist_ok=True)

        logging.info(f"Getting feature trajectories")
        all_ftrajs = get_feature_trajs(traj_top_paths, hp)

        tica, kmeans = discretize_trajectories(hp, all_ftrajs, seed)
        dtrajs = kmeans.dtrajs
        dtraj_ccs = kmeans.clustercenters

        np.save(file=str(hp_dir.joinpath('cluster_centers.npy')), arr=dtraj_ccs)
        for i in range(len(traj_top_paths['trajs'])):
            fname = Path(traj_top_paths['trajs'][i]).stem
            np.save(file=str(hp_dir.joinpath(fname)), arr=dtrajs[i])


