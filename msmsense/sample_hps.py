"""
Creates a dataframe of random hyperparameters as defined by the space in constants.py
"""
from pathlib import Path
from typing import Dict, Union, List
from argparse import ArgumentParser
import importlib.util

import numpy as np
import pandas as pd

from . import searchspace as cons

np.random.seed(149817)


def sample_hps(hp_space: Dict[str, List[Union[int, str, float]]]) -> Dict[str, Union[int, str, float]]:
    sample = {}
    for k, v in hp_space.items():
        if len(v) == 1:
            sample[k] = v[0]
        elif isinstance(v[0], float):
            sample[k] = np.random.uniform(v[0], v[1])
        elif isinstance(v[0], int):
            sample[k] = np.random.choice(np.arange(v[0], v[1]+1))
        elif isinstance(v[0], str):
            sample[k] = np.random.choice(v)
    return sample


def build_hp_sample(search_space: Dict[str, List[Union[str, int, float]]], num_trials: int) -> pd.DataFrame:
    hps = {k: [] for k in search_space.keys()}

    for i in range(num_trials):
        tmp = sample_hps(search_space)
        for k in hps.keys():
            hps[k].append(tmp[k])

    df = pd.DataFrame.from_dict(hps)
    return df


def save_sample(df: pd.DataFrame, path: Path) -> None:
    df.to_hdf(str(path), key='hyperparameters')


# TODO - replace with YAML
def get_search_space(path: Union[Path, None]):
    if path is None:
        return cons.HP_SPACE
    else:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        return module.HP_SPACE


def main(args, parser) -> None:
    if args.seed is not None:
        np.random.seed(args.seed)
    if args.search_space.suffix != '.py':
        raise ValueError('Search space must be defined as dictionary in Python file.')
    if args.output_file.suffix != '.h5':
        raise ValueError('Output file must h5')

    hp_dict = get_search_space(args.search_space)
    num_trials = args.num_trials
    out_path = args.output_file

    hp_df = build_hp_sample(hp_dict, num_trials)
    save_sample(hp_df, out_path)


def configure_parser(sub_subparser: ArgumentParser):
    p = sub_subparser.add_parser('hyperparameters')
    p.add_argument('-i', '--search-space', type=Path, help='Path to file that defines the HP search space dictionary')
    p.add_argument('-n', '--num-trials', type=int, help='Number of HP trials', default=10)
    p.add_argument('-o', '--output-file', type=Path, help='Path to hd5 file to store HP samples')
    p.add_argument('-s', '--seed', type=int, help='Random seed', default=None)
    p.set_defaults(func=main)



