from pymatgen.core.sites import PeriodicSite
from pymatgen.io.lammps.data import LammpsData, Topology

import random, copy, time
import numpy as np
import pandas as pd
from numpy.random import uniform as unif


data_file = 'compressed_555.dat'

lmp_data = LammpsData.from_file(data_file)
astr = lmp_data.structure

################## Inputs #############################
cell_size = (2, 2, 2)
n_NP = 3
latt = astr.lattice
min_dist_NP = 3 # Å vol fraction about 0.19
dia_NP = 4
# mass_fraction = 0.2 # mass fraction of NPs w.r.t PNC

#################### Make supercell ###################
"""
t1 = time.time()
astr.make_supercell(cell_size)
print ("Time taken to make {} supercell: {}".format(cell_size, time.time()-t1))
l = astr.lattice.a
print ('Made a supercell box of the polymer of size {} Å'.format(l))
"""
############### Distribute NPs in scell ################
if min_dist_NP is None:
    min_dist_NP = 2 * dia_NP + 3 # Angstroms
    print ('Setting default minimum NP distance ' \
                        'of {} Å'.format(min_dist_NP))

random_coords = []
new_loc_tries = 0
NPs_added = 0

while NPs_added < n_NP - 1 and new_loc_tries < 100: # and new_loc_tries < 1000:
    new_loc_tries += 1
    if len(random_coords) == 0:
        new_fracs = [unif(0, 1), unif(0, 1), unif(0, 1)]
        random_coords.append(new_fracs)
        continue

    # Starting from the second random coords, check the min_dist_NP constraint
    # considering the periodic boundary condition (pbc)
    added_new_fracs = False
    num_tries = 0
    while not added_new_fracs and num_tries < 500:
        num_tries += 1
        # Get the next new_fracs
        new_fracs = [unif(0, 1), unif(0, 1), unif(0, 1)]
        new_carts = latt.get_cartesian_coords(new_fracs)
        # Get points within a sphere for second point
        coords_in_new_sphere = latt.get_points_in_sphere(random_coords,
                                                new_carts, min_dist_NP)
        if len(coords_in_new_sphere) > 0:
            continue
        else:
            random_coords.append(new_fracs)
            added_new_fracs = True
            NPs_added += 1
            print ('Added new random coordinate. {} remaining..'.format(
                                                    n_NP - NPs_added))
            print (num_tries, new_loc_tries)

######## Get the molecule-IDs (chain IDs) for atoms in a sphere #########
atoms_df = lmp_data.atoms
astr = lmp_data.structure
all_fracs = astr.frac_coords
latt = astr.lattice
copy_astr = copy.deepcopy(astr)

atom_locs_to_remove = []
for center_of_NP in random_coords:
    center_of_NP = latt.get_cartesian_coords(center_of_NP)
    atoms_in_sphere =  latt.get_points_in_sphere(all_fracs, center_of_NP,
                                                            dia_NP/2)
    # The position/loc of atoms in sphere
    atom_ilocs_in_sphere = [i[2] for i in atoms_in_sphere]

    # get an array of molecule IDs for each atom ID
    mol_ids_in_sphere = np.unique([atoms_df["molecule-ID"].iloc[i] for i in \
                                                        atom_ilocs_in_sphere])

    # get all atoms with the corresponding molecule IDs
    for i in mol_ids_in_sphere:
        ith_mol_atoms = atoms_df[atoms_df["molecule-ID"]==i].index.to_list()
        ith_mol_atom_locs = [atoms_df.index.get_loc(i) for i in ith_mol_atoms]
        atom_locs_to_remove += ith_mol_atom_locs

atom_locs_to_remove = np.unique(atom_locs_to_remove)
# remove the atoms for to make all NP cavities
copy_astr.remove_sites(atom_locs_to_remove)

# add random coords as a second species
for coords in random_coords:
    props = {'charge': 0.0,
             'velocities': [unif(-1, 1), unif(-1, 1), unif(-1, 1)]}
    copy_astr.append('Li', coords, coords_are_cartesian=False, properties=props)


# function to get atom_IDs from Molecule-IDs
def get_atom_ids_from_mol_ids(molecule_ids, chain_length):
    """
    The molecule IDs are given in the order of atom IDs based on chain length
    For a chain length (n) of 25,
    Atom-IDs    Molecule-ID
    1-25        1
    26-50       2
    51-75       3
    ...
    Therefore, for molecule-ID "x" --> atom-IDs are (x-1)n+1 - (x)n
    """
    atom_ids = []
    for x in molecule_ids:
        atoms_of_mol_x = [i for i in range(
                            (x-1)*chain_length+1, x*chain_length+1)]
        atom_ids += atoms_of_mol_x

    return atom_ids

# create a new data file with the new

atoms_df = lmp_data.atoms
velocities_df = lmp_data.velocities
bonds_df = lmp_data.topology['Bonds']
angles_df = lmp_data.topology['Angles']

mol_ids_to_remove = mol_ids_in_sphere

atom_ids_to_remove = get_atom_ids_from_mol_ids(mol_ids_to_remove)

new_atom_ids = list(set(atoms_df.index.to_list()) - \
                    set(atom_ids_to_remove))

# make new lists with removed atoms, velocites, bonds and angles
new_atoms_df = atoms_df.loc[new_atom_ids]
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



# add random coords as a second species with random velocities
def get_NP_dfs(self, NP_coords, new_atoms_df):
    """
    Returns new type_2_atoms_df and type_2_velocities_df
    
    Args:
        NP_coords (array): of all the random coords
        
        new_atoms_df (dataframe): of all the atoms after cavities made
    """
    atom_ids_max = new_atoms_df.index.max()
    mol_ids_max = new_atoms_df['molecule-ID'].max()
    
    NP_atom_ids = [atom_ids_max+i+1 for i in range(len(NP_coords))]
    NP_mol_ids = [mol_ids_max+i+1 for i in range(len(NP_coords))]
    
    NP_dict = {}
    NP_dict['molecule-ID'] = NP_mol_ids
    NP_dict['type'] = np.zeros(len(NP_coords), dtype=np.int32) + 2 
    NP_dict['q'] = np.zeros(len(NP_coords))
    NP_dict['x'] = NP_coords[:, 0]
    NP_dict['y'] = NP_coords[:, 1]
    NP_dict['z'] = NP_coords[:, 2]
    NP_dict['nx'] = NP_dict['ny'] = NP_dict['nz'] = 0
    
    NP_atoms_df = pd.DataFrame(data=NP_dict, index=NP_atom_ids)
    
    random_velocities = np.array([[unif(-1, 1), unif(-1, 1), unif(-1, 1)] \
                                     for i in range(len(NP_coords))])
    
    NP_velocities_dict = {}
    NP_velocities_dict['vx'] = random_velocities[:, 0]
    NP_velocities_dict['vy'] = random_velocities[:, 1]
    NP_velocities_dict['zy'] = random_velocities[:, 2]
    
    NP_velocities_df = pd.DataFrame(data=NP_velocities_dict, index=NP_atom_ids)
    
    return NP_atoms_df , NP_velocities_df

for coords in random_coords:
    props = {'charge': 0.0,
             'velocities': [unif(-1, 1), unif(-1, 1), unif(-1, 1)]}
    


new_lmp_data = LammpsData(lmp_data.box, lmp_data.masses, new_atoms_df, 
                          velocities=new_velocities_df, topology=new_topo, 
                          force_field=lmp_data.force_field, atom_style='full')


### convert lammps data file to a trajectory (or dump)



## problem: NP_coords not converted to box lattice fractional coordinates
"""
Solution:
convert all new_atoms_df coords to fractional coords
attach the NP fractional coords
convert to cartesian coords
create a new new_atoms_df with converted cartesian coords

make a new box with updated bounds
create new_lmp_data




"""








# cc
