"""
Creates a dataframe of random hyperparameters as defined by the space in constants.py
"""
from pathlib import Path
from typing import Dict, Union, List
from argparse import ArgumentParser
import yaml
from copy import deepcopy

from hyperopt.pyll.stochastic import sample
from hyperopt import hp
import numpy as np
import pandas as pd

# def sample_hps(hp_space: Dict[str, List[Union[int, str, float]]]) -> Dict[str, Union[int, str, float]]:
#     sample = {}
#     for k, v in hp_space.items():
#         if len(v) == 1:
#             sample[k] = v[0]
#         elif isinstance(v[0], float):
#             sample[k] = np.random.uniform(v[0], v[1])
#         elif isinstance(v[0], int):
#             sample[k] = np.random.choice(np.arange(v[0], v[1]+1))
#         elif isinstance(v[0], str):
#             sample[k] = np.random.choice(v)
#     return sample
#

template = dict(
    cluster__max_iter=None,
    cluster__stride=None,
    tica__dim=None,
    tica__lag=None,
    tica__kinetic_map=None,
    tica__stride=None,
    cluster__k=None,
    feature__value=None,
    dihedrals__which=None,
    distances__scheme=None,
    distances__transform=None,
    distances__steepness=0,
    distances__centre=0,
)

SEARCH_SPACE = {'cluster__max_iter': 1000,
                 'cluster__stride': 10, 
                 'tica__dim': hp.quniform('tica__dim', 1, 20, 1), 
                 'tica__lag': hp.quniform('tica__lag', 1, 100, 1), 
                 'tica__kinetic_map': True, 
                 'tica__stride': 1, 
                 'cluster__k': hp.quniform('cluster__k', 10, 500, 1), 
                 'feature__value': hp.choice('feature__value', [
                    {'feature__value': 'dihedrals', 
                     'dihedrals__which': 'all'},
                    {'feature__value': 'distances', 
                     'distances__scheme': hp.choice('distances__scheme', ['ca', 'closest-heavy']), 
                     'distances__transform': hp.choice('distances__transform', [
                        'linear', 
                        {'distances__transform': 'logistic', 
                         'distances__steepness': hp.uniform('distances__steepness', 0.1, 50), 
                         'distances__centre': hp.uniform('distances__centre', 0.2, 1.5)}])}
                    ])
                }


def flatten(d):
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            items.extend(flatten(v).items())
        else:
            items.append((k, v))
    return dict(items)


def build_hp_sample(num_trials: int) -> pd.DataFrame:
    from addict import Dict
    all_hps = [Dict(deepcopy(template)) for _ in range(num_trials)]
    for i,  hp in enumerate(all_hps):
        hp.update(flatten(dict(sample(SEARCH_SPACE))))
    df = pd.concat([pd.DataFrame.from_dict({k: [v] for k, v in hp.items()}) for hp in all_hps])
    for col in df.columns:
        if 'float' in str(df[col].dtype):
            if 'distance' in col:
                pass
            else:
                df[col] = df[col].astype(int)
    return df


def save_sample(df: pd.DataFrame, path: Path) -> None:
    df.to_hdf(str(path), key='hyperparameters', format='table')


def parse_yaml(path: Path) -> Dict:
    with path.open('rt') as f: data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def get_search_space(path: Union[Path, None]):
    if path is None:
        thisdir = Path(__file__).parent
        return parse_yaml(thisdir.joinpath('searchspace.yaml'))
    else:
        return parse_yaml(path)


def main(args, parser) -> None:
    if args.output_file.suffix != '.h5':
        raise ValueError('Output file must h5')

    num_trials = args.num_trials
    out_path = args.output_file

    hp_df = build_hp_sample(num_trials)
    hp_df.reset_index(inplace=True)
    save_sample(hp_df, out_path)


def configure_parser(sub_subparser: ArgumentParser):
    p = sub_subparser.add_parser('hyperparameters')
    p.add_argument('-n', '--num-trials', type=int, help='Number of HP trials', default=10)
    p.add_argument('-o', '--output-file', type=Path, help='Path to hd5 file to store HP samples',
                   required=True)
    p.set_defaults(func=main)



