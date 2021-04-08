from importlib import reload
import structure

# temp
from pymatgen import Structure, Lattice, Element, Composition
from pymatgen.io.lammps.data import LammpsData, LammpsBox

import random, copy, time
import numpy as np
import pandas as pd
from numpy.random import uniform as unif

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
polymer_parameters['label'] = reg_id.create_id()

# Create polymer object
poly = polymer(polymer_parameters)

astr = poly.lmp_data.structure
latt = astr.lattice
# get frac_coords according to the astr lattice
fracs_1 = astr.frac_coords
# get atom_ids list
atoms_df_1 = poly.lmp_data.atoms
atom_ids_1 = atoms_df_1.index.to_list()
# get atom_ids of atoms with type 2
type2_atom_ids_1 = atoms_df_1[atoms_df_1['type']==2].index.to_list()
# get bonds and angles dfs
bonds_df_1 = poly.lmp_data.topology['Bonds']
angles_df_1 = poly.lmp_data.topology['Angles']
# get molecule_ids list
mol_ids_1 = poly.lmp_data.atoms['molecule-ID'].to_list()

# get NP_coords
NP_fracs = poly.get_NP_locs()
# convert NP_Coords to carts
NP_carts = latt.get_cartesian_coords(NP_fracs)
# and make cavities in fracs_1
atom_indices_to_remove = []
for center_of_NP in NP_carts:
    atoms_in_sphere =  latt.get_points_in_sphere_py(fracs_1,
                                        center_of_NP, poly.dia_NP/2)
    print ('No. of atoms in sphere: {}'.format(len(atoms_in_sphere)))
    # The position/loc of atoms in sphere
    atom_indices_in_sphere = [i[2] for i in atoms_in_sphere]
    atom_indices_to_remove += atom_indices_in_sphere
    
atom_indices_to_remove = np.unique(atom_indices_to_remove)


# get fracs, atom_ids and mol_ids for atoms out of these cavities
frac_indices_with_cavities = set([i for i in range(len(fracs_1))]) - \
                                    set(atom_indices_to_remove)
fracs_with_cavities, atom_ids_with_cavities,mol_ids_with_cavities = [], [], []
for i in frac_indices_with_cavities:
    fracs_with_cavities.append(fracs_1[i])
    atom_ids_with_cavities.append(atom_ids_1[i])
    mol_ids_with_cavities.append(mol_ids_1[i])

new_astr = Structure(latt, ['H' for i in range(len(fracs_with_cavities))], 
                     fracs_with_cavities, coords_are_cartesian=False)

# NOTE: These atom indices to remove will also give the mol_ids to remove
mol_ids_to_remove = [mol_ids_1[i] for i in atom_indices_to_remove]
mol_ids_to_remove = np.unique(mol_ids_to_remove)

# get indices from mol_ids_with_cavities for mol_ids_to_remove if any
mol_indices_to_remove = [i for i, mol_id in enumerate(mol_ids_with_cavities) \
                               if mol_id in mol_ids_to_remove]
mol_indices_to_keep = set([i for i in range(len(mol_ids_with_cavities))]) - \
                               set(mol_indices_to_remove)
# get fracs, atom_ids and mol_ids for atoms with removed mol_ids 
# that are connected inside cavities
disconnected_fracs, disconnected_atom_ids, disconnected_mol_ids = [], [], []
for i in mol_indices_to_keep:
    disconnected_fracs.append(fracs_with_cavities[i])
    disconnected_atom_ids.append(atom_ids_with_cavities[i])
    disconnected_mol_ids.append(mol_ids_with_cavities[i]) 
    
# TEST: check cavities
for carts in NP_carts:
    print (len(latt.get_points_in_sphere_py(new_astr.frac_coords, carts, 
                                            poly.dia_NP/2)))

# get_atoms_df from latt, fracs, atom_ids, mol_ids
def get_new_atoms_df(latt, fracs, atom_ids, mol_ids, 
                     type_int=1, charge=0, NP=False, return_topo=True, 
                     bonds_df_1=None, angles_df_1=None):
    if NP == True: # assign new atom ids and mol ids 
        atom_ids = [max(atom_ids)+i+1 for i in range(len(fracs))]
        mol_ids = [max(mol_ids)+i+1 for i in range(len(fracs))]
    
    # remove all pre-existing type 2 atom_ids
    rem_indices = [i for i, atom_id in enumerate(atom_ids) \
                      if atom_id in type2_atom_ids_1]
    
    keep_indices = set([i for i in range(len(fracs))]) - set(rem_indices)
    
    fracs_2, atom_ids_2, mol_ids_2 = [], [], []
    for i in keep_indices:
        fracs_2.append(fracs[i])
        atom_ids_2.append(atom_ids[i])
        mol_ids_2.append(mol_ids[i])
        
    atoms_dict = {}
    atoms_dict['molecule-ID'] = mol_ids_2
    atoms_dict['type'] = np.zeros(len(mol_ids_2), dtype=np.int16) + type_int
    atoms_dict['q'] = np.zeros(len(mol_ids_2)) + charge
    
    # convert fracs to carts
    carts_2 = latt.get_cartesian_coords(fracs_2)
    atoms_dict['x'] = carts_2[:, 0]
    atoms_dict['y'] = carts_2[:, 1]
    atoms_dict['z'] = carts_2[:, 2]
    # TODO: not sure if this is necessary to be maintained
    atoms_dict['nx'] = atoms_dict['ny'] = atoms_dict['nz'] = 0
    
    atoms_df = pd.DataFrame(data=atoms_dict, index=atom_ids_2)
    
    if not return_topo:
        return atoms_df
    else:
        bonds_df_2 = bonds_df_1[bonds_df_1['atom1'].isin(atom_ids_2)]
        bonds_df_2 = bonds_df_2[bonds_df_2['atom2'].isin(atom_ids_2)]
        
        angles_df_2 = angles_df_1[angles_df_1['atom1'].isin(atom_ids_2)]
        angles_df_2 = angles_df_2[angles_df_2['atom2'].isin(atom_ids_2)]
        angles_df_2 = angles_df_2[angles_df_2['atom3'].isin(atom_ids_2)]
        
        topo = {'Bonds': bonds_df_2, 'Angles': angles_df_2}
        
    return atoms_df, topo
    

# get atoms_df_only_cavities
atoms_df_only_cavities, topo_atom_cavities = get_new_atoms_df(
                                                 latt, fracs_with_cavities, 
                                                 atom_ids_with_cavities, 
                                                 mol_ids_with_cavities, 
                                                 return_topo=True,
                                                 bonds_df_1=bonds_df_1,
                                                 angles_df_1=angles_df_1)

# get atoms_df_disconnected_cavities
atoms_df_disconnected_cavities, topo_mol_cavities = get_new_atoms_df(
                                                      latt, disconnected_fracs, 
                                                      disconnected_atom_ids, 
                                                      disconnected_mol_ids,
                                                      return_topo=True,
                                                      bonds_df_1=bonds_df_1,
                                                      angles_df_1=angles_df_1)

# get NP_atoms_df 
NP_atoms_df = get_new_atoms_df(latt, NP_fracs, atom_ids_with_cavities, 
                               mol_ids_with_cavities, type_int=2, NP=True, 
                               return_topo=False)

# create PNC with atoms_df and NP_atoms_df
def create_PNC(poly_atoms_df, NP_atoms_df, topo):
    a = latt.a
    new_lmp_box = LammpsBox([[0, a], [0, a], [0, a]])
    # Note: masses, force_field, atom_style same as that of lmp_data
    pnc_atoms_df = pd.concat([poly_atoms_df, NP_atoms_df])
    # assign random velocities
    # TODO: any better method to assign random velocities
    velocities_df = poly.lmp_data.velocities
    rand_ints = random.sample(range(1, len(poly.lmp_data.atoms.x)), 
                              len(pnc_atoms_df.x))
    velocities_df = velocities_df.loc[rand_ints]
    
    # create pnc
    pnc_lmp_data = LammpsData(new_lmp_box, poly.lmp_data.masses,
                              pnc_atoms_df, velocities=velocities_df, 
                              topology=topo, force_field=poly.lmp_data.force_field,
                              atom_style='full')
    
    return pnc_lmp_data

# create PNC with atoms only cavities
pnc_atoms_cavities = create_PNC(atoms_df_only_cavities, NP_atoms_df,
                                topo_atom_cavities)
pnc_atoms_cavities.structure.to(filename='POSCAR_pnc_atoms_cavs.vasp')
# create PNC with molecules (chains) cavities
pnc_mol_cavities = create_PNC(atoms_df_disconnected_cavities, NP_atoms_df,
                              topo_mol_cavities)
pnc_mol_cavities.structure.to(filename='POSCAR_pnc_mol_cavs.vasp')











#