from importlib import reload
import structure

structure = reload(structure)

polymer = structure.polymer

# Input parameters
n_NP = 5
dia_NP = 20
min_dist_NP = 23
chain_length = 25

# generate label
reg_id = structure.register_id()

polymer_parameters = {}
polymer_parameters['data_file'] = 'compressed_555.dat'
polymer_parameters['n_NP'] = n_NP
polymer_parameters['dia_NP'] = dia_NP
polymer_parameters['min_dist_NP'] = min_dist_NP
polymer_parameters['chain_length'] = chain_length

# Create polymer object
poly = polymer(polymer_parameters, reg_id)

# get NP_coords
NP_fracs = poly.get_NP_locs()

# get indices of frac coords in the NP cavities
cavity_atom_indices = poly.get_cavity_atom_indices(NP_fracs)

fracs_with_cavities, atom_ids_with_cavities, mol_ids_with_cavities = \
                        poly.get_non_cavity_atoms(cavity_atom_indices)

disconnected_fracs, disconnected_atom_ids, disconnected_mol_ids =\
                        poly.get_non_cavity_chain_atoms(cavity_atom_indices)

# get atoms_df_only_cavities
atoms_df_only_cavities, topo_atom_cavities = poly.get_new_atoms_df(
                                                 fracs_with_cavities, 
                                                 atom_ids_with_cavities, 
                                                 mol_ids_with_cavities, 
                                                 return_topo=True)
# get atoms_df_disconnected_cavities
atoms_df_disconnected_cavities, topo_mol_cavities = poly.get_new_atoms_df(
                                                      disconnected_fracs, 
                                                      disconnected_atom_ids, 
                                                      disconnected_mol_ids,
                                                      return_topo=True)
# get NP_atoms_df 
NP_atoms_df = poly.get_new_atoms_df(NP_fracs, atom_ids_with_cavities, 
                               mol_ids_with_cavities, type_int=2, NP=True, 
                               return_topo=False)

# create PNC with atoms only cavities
pnc_atoms_cavities = poly.create_PNC(atoms_df_only_cavities, NP_atoms_df,
                                     topo_atom_cavities)

# write POSCAR, LAMMPS_data_file, trajectory
# NOTE: trajectory file is used as input for soq code in C++ (by Jan Michael)
pnc_atoms_cavities.structure.to(
                filename='POSCAR_pnc_atoms_{}.vasp'.format(poly.label))
pnc_atoms_cavities.write_file('pnc_atoms_{}.dat'.format(poly.label))
# the filename for soq should be of the following format: dump.000000000.txt
# TODO: change when using with soq code
traj_filename =  'atom_cavs.{:09d}.txt'.format(poly.label)
poly.write_trajectory(pnc_atoms_cavities, traj_filename)

# create PNC with molecules (chains) cavities
pnc_mol_cavities = poly.create_PNC(atoms_df_disconnected_cavities, NP_atoms_df,
                                   topo_mol_cavities)
pnc_mol_cavities.structure.to(
                filename='POSCAR_pnc_mol_{}.vasp'.format(poly.label))
pnc_mol_cavities.write_file('pnc_mol_{}.dat'.format(poly.label))
traj_filename =  'mol_cavs.{:09d}.txt'.format(poly.label)
poly.write_trajectory(pnc_mol_cavities, traj_filename)























##