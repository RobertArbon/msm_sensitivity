# MSM sensitivity  (`msmsense`)

This is a package for bootstrapping Markov state model observables from molecular dynamics simulations. It should be used alongside [MSM sensitivity analysis (`sensetools`)](https://github.com/RobertArbon/msm_sensitivity_analysis) to recreate the data for the paper [Sensitivity and optimisation of Markov state models for biomolecular systems](https://github.com/RobertArbon/MSM-Hyperparameter-Optimisation).  

## Installation

1. Create a `conda`/`mamba` environment and `cd` to root of this repo. 
2. `mamba install numpy pandas pyyaml mdtraj pyemma scipy click -y`
3. `pip install hyperopt`
4. `pip install -e . `

## Use

See the [README](https://github.com/RobertArbon/msm_sensitivity_analysis/blob/main/README.md) of [`sensetools`](https://github.com/RobertArbon/msm_sensitivity_analysis) for practical use.

To see the available commands simply use the help function on the cli: 

`$ msmsense --help`

To change the limits of the hyperparameter search space edit the `msmsense/searchspace.yaml` file. 
