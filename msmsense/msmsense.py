from pathlib import Path

import click

from .bootstrap import bootstrap

@click.group()
def cli():
    pass

@cli.command()
@click.option('-i', '--hp-sample', type=Path, help='Path to file that contains the hyperparameter samples')
@click.option('-d', '--data-dir', type=Path, help='Base directory used to determine trajectory and topology paths')
@click.option('-t', '--topology-path', type=Path, help='Topology path')
@click.option('-g', '--trajectory-glob', type=str, help='Trajectory glob string relative to --data-dir')
@click.option('-r', '--num-repeats', type=int, help='Number of bootstrap samples')
@click.option('-n', '--num-cores', type=int, help='Number of cpu cores to use.', default=1)
@click.option('-l', '--lags', type=str, help='Lags as a Python range specification start:end:stride',default='2:51:2')
@click.option('-o', '--output-dir', type=Path, help='Path to output directory')
@click.option('-s', '--seed', type=int, help='Random seed', default=None)
@click.argument('hp-ixs', type=int, nargs=-1)
def score(hp_sample, data_dir, topology_path, trajectory_glob, num_repeats, num_cores, lags, output_dir, seed, hp_ixs):
    print(hp_ixs)

    # bootstrap(hp_sample, data_dir, topology_path, trajectory_glob, num_repeats, num_cores, lags, output_dir, seed)
