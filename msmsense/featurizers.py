from typing import Optional, List, Union, Dict
import logging

import mdtraj as md
import numpy as np
from mdtraj import compute_dihedrals
from mdtraj.geometry.dihedral import indices_chi1, indices_chi2, indices_chi3, indices_chi4, indices_chi5, \
    indices_omega, indices_psi, indices_phi


def _dihedral_indices(top: md.Topology, which: str) -> np.ndarray:
    indices = [indices_phi(top),
               indices_psi(top)]
    if which == 'all':
        indices.extend([indices_chi1(top),
                         indices_chi2(top),
                         indices_chi3(top),
                         indices_chi4(top),
                         indices_chi5(top)])
    indices = np.vstack(indices)
    assert indices.shape[1] == 4
    return indices


def _dihedrals(traj: md.Trajectory, indices: np.ndarray) -> np.ndarray:
    res = compute_dihedrals(traj, indices)
    rad = np.concatenate([np.cos(res), np.sin(res)], axis=1)
    return rad


def dihedrals(traj_top_dict: Dict[str, List[str]], which: Optional[str] = 'all') -> List[np.ndarray]:
    logging.info(f"Creating dihedral trajectories using {which} torsions")
    top = md.load(traj_top_dict['top']).topology
    indices = _dihedral_indices(top, which)
    logging.info(f"Number of raw dimensions: {indices.shape[0]}")
    ftrajs = []
    for traj_path in traj_top_dict['trajs']:
        traj = md.load(traj_path,top=top)
        ftrajs.append(_dihedrals(traj, indices))
    return ftrajs


def _distances(traj: md.Trajectory, scheme: str, transform: str,
               centre: Union[float, None],  steepness: Union[float, None]):
    feat, ix = md.compute_contacts(traj, contacts='all', scheme=scheme)
    if transform == 'logistic':
        assert (centre is not None) and (steepness is not None)
        tmp = 1.0/(1.+np.exp((-1)*steepness*(feat-centre)))
        assert np.allclose(tmp.shape, feat.shape)
        feat = tmp
    return feat


def distances(traj_top_dict: Dict[str, List[str]], scheme: Optional[str] = 'closest-heavy',
              transform: Optional[str] = 'linear', centre: Optional[Union[float, None]] = None,
              steepness: Optional[Union[float, None]] = None) -> List[np.ndarray]:
    logging.info(f"Creating distance trajectories using scheme: {scheme}, with {transform} transform")
    if transform == 'logistic':
        logging.info(f"centre: {centre}, steepness: {steepness}")
    # Check for silly dimensions
    traj = md.load(traj_top_dict['trajs'][0], top=traj_top_dict['top'])
    test = _distances(traj, scheme, transform, centre, steepness)
    if test.shape[1] > 1000 and ('closest' in scheme):
        logging.info("Closest scheme is memory intensive. Consider different scheme.")
    elif test.shape[1] > 5000:
        logging.info("Large number of contact distances. Consider changing number of contacts")
    else:
        logging.info(f"Number of dimensions: {test.shape[1]}")
    ftrajs = []
    for traj_path in traj_top_dict['trajs']:
        traj = md.load(traj_path, top=traj_top_dict['top'])
        ftrajs.append(_distances(traj, scheme, transform, centre, steepness))
    return ftrajs

# def side_sidechain_torsions(reader: DataSource) -> DataSource:
#     featurizer = reader.featurizer
#     top = featurizer.topology
#     indices = np.vstack((indices_chi1(top),
#                          indices_chi2(top),
#                          indices_chi3(top),
#                          indices_chi4(top),
#                          indices_chi5(top),
#                          indices_omega(top)))
#     assert indices.shape[1] == 4
#
#     def compute_side_chains(traj):
#         res = compute_dihedrals(traj, indices)
#         # cossin
#         rad = np.dstack((np.cos(res), np.sin(res)))
#         rad = rad.reshape(rad.shape[0], rad.shape[1] * rad.shape[2])
#         print('shape chunk:',rad.shape)
#         return rad
#
#     featurizer.add_custom_func(compute_side_chains, dim=len(indices)*2)
#     return reader
#
#
# def add_phipsi_dihdedrals(reader: DataSource) -> DataSource:
#     reader.featurizer.add_backbone_torsions(cossin=True)
#     return reader
#
#
# def num_contacts(reader: DataSource) -> int:
#     traj = md.load_frame(reader.filenames[0], top=reader.featurizer.topology, index=0)
#     _, ix = md.compute_contacts(traj, contacts='all')
#     return ix.shape[0]
#
#
# def add_contacts(reader: DataSource, cutoff: float, scheme: Optional[str] = 'closest-heavy') -> DataSource:
#     dim = num_contacts(reader)
#
#     def _contacts(traj):
#         feat, ix = md.compute_contacts(traj, contacts='all', scheme=scheme)
#         feat = ((feat <= cutoff)*1).astype(np.float32)
#         return feat
#
#     reader.featurizer.add_custom_func(_contacts, dim=dim)
#     return reader
