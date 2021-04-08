from importlib import reload
import structure

# temp
from pymatgen import Structure, Lattice, Element, Composition
from pymatgen.io.lammps.data import LammpsData, LammpsBox

import random, copy, time
import numpy as np
import pandas as pd
from numpy.random import uniform as unif
### end temp

structure = reload(structure)

polymer = structure.polymer

# Input parameters
n_NP = 3
dia_NP = 2.5
min_dist_NP = 4.5
chain_length = 25

# generate label
reg_id = structure.register_id()

polymer_parameters = {}
polymer_parameters['data_file'] = 'compressed.dat'
polymer_parameters['n_NP'] = n_NP
polymer_parameters['dia_NP'] = dia_NP
polymer_parameters['min_dist_NP'] = min_dist_NP
polymer_parameters['chain_length'] = chain_length
polymer_parameters['label'] = reg_id.create_id()

# Create polymer object
poly = polymer(polymer_parameters)
# get random coords for the NPs in the polymer matrix
random_coords = poly.get_NP_locs()

######################### general variables ###########################
lmp_data = poly.lmp_data
astr = lmp_data.structure
latt = lmp_data.structure.lattice

# write only NP_locs to poscar
np_astr = Structure(latt, ['H' for i in range(len(random_coords))], 
                    random_coords, coords_are_cartesian=False)
np_astr.to(filename='POSCAR_NPs')

atoms_df = lmp_data.atoms
velocities_df = lmp_data.velocities
bonds_df = lmp_data.topology['Bonds']
angles_df = lmp_data.topology['Angles']

#all_fracs_2 from atoms_df
atoms_carts = np.array([atoms_df.x, atoms_df.y, atoms_df.z]).T
all_fracs = latt.get_fractional_coords(atoms_carts)

######################### get_mol_ids_to_remove  ###########################
# remove atom_ids and make a POSCAR
# get the mol_ids_to_remove to make cavities for the NPs
# mol_ids_to_remove = poly.get_mol_ids_to_remove(random_coords)
mol_ids_to_remove = []
for center_of_NP in random_coords:
    center_of_NP = latt.get_cartesian_coords(center_of_NP)
    atoms_in_sphere =  latt.get_points_in_sphere(all_fracs,
                                        center_of_NP, poly.dia_NP/2)
    print ('No. of atoms in sphere: {}'.format(len(atoms_in_sphere)))
    # The position/loc of atoms in sphere
    atom_ilocs_in_sphere = [i[2] for i in atoms_in_sphere]

    # get an array of molecule IDs for each atom ID
    mol_ids_in_sphere = np.unique([atoms_df["molecule-ID"].iloc[i] \
                                for i in atom_ilocs_in_sphere])
    mol_ids_to_remove += list(mol_ids_in_sphere)

mol_ids_to_remove = np.unique(mol_ids_to_remove)

######################### get_atom_ids_to_remove ###########################
atom_ids_to_remove = poly.get_atom_ids_from_mol_ids(mol_ids_to_remove)

######################### make_data_file ###########################
# new_atom_ids
new_atom_ids = list(set(atoms_df.index.to_list()) - set(atom_ids_to_remove))

new_atoms_df = atoms_df.loc[new_atom_ids]


#### testing structure with cavities
# new_atoms_coords = np.array([new_atoms_df.x, new_atoms_df.y, new_atoms_df.z]).T
# new_astr = Structure(latt, ['H' for i in range(len(new_atoms_coords))], 
#                      new_atoms_coords, coords_are_cartesian=True)
#new_astr.to(filename='POSCAR_w_cavities')


# velocites and other properties
new_velocities_df = velocities_df.loc[new_atom_ids]
new_bonds_df = bonds_df[~bonds_df['atom1'].isin(atom_ids_to_remove)]
new_bonds_df = new_bonds_df[~new_bonds_df['atom2'].isin(
                                                atom_ids_to_remove)]

new_angles_df = angles_df[~angles_df['atom1'].isin(atom_ids_to_remove)]
new_angles_df = new_angles_df[~new_angles_df['atom2'].isin(
                                                atom_ids_to_remove)]
new_angles_df = new_angles_df[~new_angles_df['atom3'].isin(
                                                atom_ids_to_remove)]

new_topo = {'Bonds': new_bonds_df, 'Angles': new_angles_df}

NP_coords = random_coords
# convert NP coords to cartesian before adding
NP_coords = latt.get_cartesian_coords(NP_coords)
NP_atoms_df, NP_velocities_df = poly.get_NP_dfs(
                                np.array(NP_coords), new_atoms_df)

new_atoms_df = pd.concat([new_atoms_df, NP_atoms_df])
new_velocities_df = pd.concat([new_velocities_df, NP_velocities_df])

new_lmp_data = LammpsData(lmp_data.box, lmp_data.masses, new_atoms_df,
                                  velocities=new_velocities_df,
                                  topology=new_topo,
                                  force_field=lmp_data.force_field,
                                  atom_style='full')

# new_lmp_data.structure.to(filename='POSCAR_test_3.vasp')

# test if this is successful
ss = new_lmp_data.structure
nps = ss.cart_coords[-5:]
print ('carts: {}'.format(nps))
for carts in nps:
    xx = ss.lattice.get_points_in_sphere(ss.frac_coords, carts, 
                                         poly.dia_NP/2)
    print (xx[2])
    #print (len(xx))
    



#########################  ###########################








##