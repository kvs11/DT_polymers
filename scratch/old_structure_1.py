#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: chaitanyakolluru
"""

"""
Make polymer nano-composite (PNC) structures

Read a polymer structure from file
Make supercell

Get parameters regarding nano-particles (NPs)

> minimum diameter, maximum diameter
> number of NPs
> type of distribution of NPs in the box (--> see MakeSimBox code)
> mass fraction of NPs (?)

-------------

> Identify the beads that are in each sphere
> Remove the entire polymer chains that are linked to the beads wihin the sphere

NOTE: Find how to identify the entire chain from the atom beads and then remove them.

-------------

> Make a new data file with the NPs
> Calculate S(Q) and compare

-------------

"""

# from pymatgen import Structure, Lattice, Element, Composition
from pymatgen.io.lammps.data import LammpsData, LammpsBox

import random
import numpy as np
import pandas as pd
from numpy.random import uniform as unif

# same as in Fantastx
class register_id(object):
    def __init__(self):
        self.label = 0
    def create_id(self):
        self.label += 1
        return self.label

class polymer(object):
    """
    This class contains all the attributes of a polymer model. Creating random 
    polymer models, or performing GA/mutation operations on a model

    It also reads the intial polymer cell and stores it's supercell. Later, 
    using this cell, multiple models can be generated.

    The purpose of this class in big picture is to make structures which would 
    then be evaluated for S(Q). i.e., velocities and topology (bonds & angles) 
    is not important for this part.
    """
    def __init__(self, polymer_parameters, reg_id):
        """
        Args:

        polymer_parameters (dict): A dictionary with path to a polymer matrix
        (supercell) LAMMPS data file, number of nanoparticles (NPs) needed,
        minimum distance constraint for the NPs and diameter of NPs
        dicttioanry keys --> ['data_file', 'n_NP', 'min_dist_NP', 'dia_NP']
        """
        # path to data_file
        if 'data_file' not in polymer_parameters:
            print ('Error: Please provide path to data_file.')
        data_file = polymer_parameters['data_file']
        self.lmp_data = LammpsData.from_file(data_file)

        # number of NPs
        if 'n_NP' not in polymer_parameters.keys():
            self.n_NP = 10
            print ('Setting default no. of NPs in polymer matrix to 10')
        else:
            self.n_NP = polymer_parameters['n_NP']

        # diameter of NP in Å
        if 'dia_NP' not in polymer_parameters:
            self.dia_NP = 10
            print ('Setting default diameter of NP to 10 Å')
        else:
            self.dia_NP = polymer_parameters['dia_NP']

        # minimum distance between any two NPs
        if 'min_dist_NP' not in polymer_parameters.keys():
            self.min_dist_NP = self.dia_NP + 3 # Angstroms
            print ('Setting default minimum NP distance '
                                'of {} Å'.format(self.min_dist_NP))
        else:
            self.min_dist_NP = polymer_parameters['min_dist_NP']

        self.chain_length = polymer_parameters['chain_length']

        self.label = reg_id.create_id()
        
        # attributes derived from lmp_data --> for convenience
        atoms_df_1 = self.lmp_data.atoms
        self.atoms_df_1 = atoms_df_1
        self.type2_atom_ids_1 = \
                    atoms_df_1[atoms_df_1['type']==2].index.to_list()
        self.bonds_df_1 = self.lmp_data.topology['Bonds']
        self.angles_df_1 = self.lmp_data.topology['Angles']
        self.mol_ids_1 = self.lmp_data.atoms['molecule-ID'].to_list()            
        self.astr = self.lmp_data.structure
        self.latt = self.astr.lattice
        self.fracs_1 = self.astr.frac_coords

        # Any other parameters needed should be saved here from 
        # input parameters


    def get_NP_locs(self):
        """
        Get the fractional coordinaees of nanoparticles (NPs) within a box.
        The NPs among themselves should obey a minimum distance constraint.
        The size of each NP should be considered.

        Args:

        n_NP (int): Number of nanp particles need to be added to polymer matrix

        min_dist_NP (float): The minimum distance between two NPs. Defaults to 
        the (2 * dia_NP + 1) Å

        """
        n_NP = self.n_NP
        latt = self.lmp_data.structure.lattice
        min_dist_NP = self.min_dist_NP

        random_coords = []
        new_loc_tries = 0
        NPs_added = 0

        while NPs_added < n_NP - 1 and new_loc_tries < 100:
            new_loc_tries += 1
            if len(random_coords) == 0:
                new_fracs = [unif(0, 1), unif(0, 1), unif(0, 1)]
                random_coords.append(new_fracs)
                continue

            # Starting from second random coords, check min_dist_NP constraint
            # considering the periodic boundary condition (pbc)
            added_new_fracs = False
            num_tries = 0
            while not added_new_fracs and num_tries < 500:
                num_tries += 1
                # Get the next new_fracs
                new_fracs = [unif(0, 1), unif(0, 1), unif(0, 1)]
                new_carts = latt.get_cartesian_coords(new_fracs)
                # Get points within a sphere for second point
                coords_in_new_sphere = latt.get_points_in_sphere_py(
                                random_coords, new_carts, min_dist_NP)
                if len(coords_in_new_sphere) > 0:
                    continue
                else:
                    random_coords.append(new_fracs)
                    added_new_fracs = True
                    NPs_added += 1
                    print ('Added new random coordinate. {} remaining..'.format(
                                                            n_NP - NPs_added))
                    print (num_tries, new_loc_tries)

        return random_coords
    
    
    def get_cavity_atom_indices(self, NP_fracs):
        """
        """
        # convert NP_Coords to carts
        NP_carts = self.latt.get_cartesian_coords(NP_fracs)
        # and make cavities in fracs_1
        cavity_atom_indices = []
        for center_of_NP in NP_carts:
            atoms_in_sphere =  self.latt.get_points_in_sphere_py(self.fracs_1,
                                        center_of_NP, self.dia_NP/2)
            print ('No. of atoms in sphere: {}'.format(
                                        len(atoms_in_sphere)))
            # The position/loc of atoms in sphere
            atom_indices_in_sphere = [i[2] for i in atoms_in_sphere]
            cavity_atom_indices += atom_indices_in_sphere
            
        return np.unique(cavity_atom_indices)


    def get_non_cavity_atoms(self, cavity_atom_indices):
        """
        """
        # get fracs, atom_ids and mol_ids for atoms out of these cavities
        frac_indices_with_cavities = set([i for i in \
                        range(len(self.fracs_1))]) - set(cavity_atom_indices)
        fracs_with_cavities, atom_ids_with_cavities, mol_ids_with_cavities = \
                                            [], [], []
        for i in frac_indices_with_cavities:
            fracs_with_cavities.append(self.fracs_1[i])
            atom_ids_with_cavities.append(self.atom_ids_1[i])
            mol_ids_with_cavities.append(self.mol_ids_1[i])        
        
        return fracs_with_cavities, atom_ids_with_cavities, \
                    mol_ids_with_cavities
               
                    
    def get_non_cavity_chain_atoms(self, cavity_atom_indices):
        """
        """
        fracs_with_cavities, atom_ids_with_cavities, \
                    mol_ids_with_cavities = self.get_non_cavity_atoms(
                                                    cavity_atom_indices)
                    
        # NOTE: cavity atom indices will also give the mol_ids to remove
        mol_ids_to_remove = [self.mol_ids_1[i] for i in cavity_atom_indices]
        mol_ids_to_remove = np.unique(mol_ids_to_remove)
        
        # get indices from mol_ids_with_cavities for mol_ids_to_remove if any
        mol_indices_to_remove = [i for i, mol_id in enumerate(
                        mol_ids_with_cavities) if mol_id in mol_ids_to_remove]
        mol_indices_to_keep = set([i for i in range(len(
                        mol_ids_with_cavities))]) - set(mol_indices_to_remove)
        # get fracs, atom_ids and mol_ids for atoms with removed mol_ids 
        # that are connected inside cavities
        disconnected_fracs, disconnected_atom_ids, disconnected_mol_ids = \
                                                                    [], [], []
        for i in mol_indices_to_keep:
            disconnected_fracs.append(fracs_with_cavities[i])
            disconnected_atom_ids.append(atom_ids_with_cavities[i])
            disconnected_mol_ids.append(mol_ids_with_cavities[i])
            
        return disconnected_fracs, disconnected_atom_ids, disconnected_mol_ids

    
    def get_new_atoms_df(self, fracs, atom_ids, mol_ids, type_int=1, charge=0, 
                         NP=False, return_topo=True):
        """
        """
        latt = self.latt
        bonds_df_1 = self.bonds_df_1
        angles_df_1 = self.angles_df_1
        
        if NP == True: # assign new atom ids and mol ids 
            atom_ids = [max(atom_ids)+i+1 for i in range(len(fracs))]
            mol_ids = [max(mol_ids)+i+1 for i in range(len(fracs))]
        
        # remove all pre-existing type 2 atom_ids
        rem_indices = [i for i, atom_id in enumerate(atom_ids) \
                          if atom_id in self.type2_atom_ids_1]
        
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


    def create_PNC(self, poly_atoms_df, NP_atoms_df, topo):
        """
        """
        a = self.latt.a
        new_lmp_box = LammpsBox([[0, a], [0, a], [0, a]])
        # Note: masses, force_field, atom_style same as that of lmp_data
        pnc_atoms_df = pd.concat([poly_atoms_df, NP_atoms_df])
        # assign random velocities
        # TODO: any better method to assign random velocities
        velocities_df = self.lmp_data.velocities
        rand_ints = random.sample(range(1, len(self.lmp_data.atoms.x)), 
                                  len(pnc_atoms_df.x))
        velocities_df = velocities_df.loc[rand_ints]
        
        # create pnc
        pnc_lmp_data = LammpsData(new_lmp_box, self.lmp_data.masses,
                                  pnc_atoms_df, velocities=velocities_df, 
                                  topology=topo, 
                                  force_field=self.lmp_data.force_field,
                                  atom_style='full')
        
        return pnc_lmp_data


    def get_mol_ids_to_remove(self, NP_fracs):
        """
        Returns all the molecule-IDs to be removed in order to make 
        cavities  for the new NPs located at the random_coords
        """
        atoms_df = self.lmp_data.atoms
        astr = self.lmp_data.structure
        latt = astr.lattice
        all_fracs = astr.frac_coords
        NP_carts = latt.get_cartesian_coords(NP_fracs)
        
        mol_ids_to_remove = []
        for center_of_NP in NP_carts:
            atoms_in_sphere =  latt.get_points_in_sphere_py(all_fracs,
                                            center_of_NP, self.dia_NP/2)
            # The position/loc of atoms in sphere
            atom_indices_in_sphere = [i[2] for i in atoms_in_sphere]

            # get an array of molecule IDs for each atom ID
            mol_ids_in_sphere = np.unique(
                                [atoms_df["molecule-ID"].iloc[i] \
                                 for i in atom_indices_in_sphere])
            mol_ids_to_remove += list(mol_ids_in_sphere)  
            
        return np.unique(mol_ids_to_remove)
    
    
    def get_atom_ids_from_mol_ids(self, molecule_ids):
        """
        The molecule IDs are given in the order of atom IDs based on
        chain length. For a chain length (n) of 25,

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
                    (x-1)*self.chain_length+1, x*self.chain_length+1)]
            atom_ids += atoms_of_mol_x
            
        # Also add the atoms of type 2 to remove those atom_ids as well
        atom_ids += self.type_2_ids

        return np.unique(atom_ids)


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
        NP_dict['type'] = np.zeros(len(NP_coords), dtype=np.int16) + 2 
        NP_dict['q'] = np.zeros(len(NP_coords))
        NP_dict['x'] = NP_coords[:, 0]
        NP_dict['y'] = NP_coords[:, 1]
        NP_dict['z'] = NP_coords[:, 2]
        NP_dict['nx'] = NP_dict['ny'] = NP_dict['nz'] = 0
        
        NP_atoms_df = pd.DataFrame(data=NP_dict, index=NP_atom_ids)
        
        random_velocities = np.array([[unif(-1, 1), unif(-1, 1), unif(-1, 1)]\
                                         for i in range(len(NP_coords))])
        
        NP_velocities_dict = {}
        NP_velocities_dict['vx'] = random_velocities[:, 0]
        NP_velocities_dict['vy'] = random_velocities[:, 1]
        NP_velocities_dict['zy'] = random_velocities[:, 2]
        
        NP_velocities_df = pd.DataFrame(data=NP_velocities_dict, 
                                        index=NP_atom_ids)
        
        return NP_atoms_df , NP_velocities_df


    def make_data_file(self, mol_ids_to_remove, NP_coords):
        """
        Function to
        1.
        Remove the atoms from the lmp_data
        Add the random NPs to the lmp_data
        create a new data file
        Use the new_data_file as input to get the SOQ & for LAMMPS relaxation

        OR

        2. Use the rand_PNC_astr
        convert it to new_data_file
        Use the new_data_file as input to get the SOQ & for LAMMPS 
        relaxation
        """
        lmp_data = self.lmp_data
        latt = self.lmp_data.structure.lattice
        atoms_df = lmp_data.atoms
        velocities_df = lmp_data.velocities
        bonds_df = lmp_data.topology['Bonds']
        angles_df = lmp_data.topology['Angles']

        atom_ids_to_remove = self.get_atom_ids_from_mol_ids(
                                                mol_ids_to_remove)

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
        
        
        # convert NP coords to cartesian before adding
        NP_coords = latt.get_cartesian_coords(NP_coords)
        NP_atoms_df, NP_velocities_df = self.get_NP_dfs(
                                        np.array(NP_coords), new_atoms_df)
        
        new_atoms_df = pd.concat([new_atoms_df, NP_atoms_df])
        new_velocities_df = pd.concat([new_velocities_df, NP_velocities_df])
        
        # convert all new_atoms_df coords to fractional coords
        # old_frac_coords = latt.get_fractional_coords(np.array([new_atoms_df.x, 
        #                                                new_atoms_df.y,
        #                                                new_atoms_df.z]).T)
        # attach the NP fractional coords
        # new_frac_coords = np.concatenate([old_frac_coords, NP_coords])
        # convert to cartesian coords
        # new_cart_coords = latt.get_cartesian_coords(new_frac_coords)
        # create a combined new_atoms_df by adding NP_atoms_df
        new_atoms_df = pd.concat([new_atoms_df, NP_atoms_df])
        # update the x, y, z coordinates in this with new cartesian coords
        # new_atoms_df.x = new_cart_coords[:, 0]
        # new_atoms_df.y = new_cart_coords[:, 1]
        # new_atoms_df.z = new_cart_coords[:, 2]

        # create new_lmp_data        
        new_lmp_data = LammpsData(lmp_data.box, lmp_data.masses, new_atoms_df,
                                  velocities=new_velocities_df,
                                  topology=new_topo,
                                  force_field=lmp_data.force_field,
                                  atom_style='full')

        # write the new_lmp_data to a file
        new_lmp_data.write_file('pnc_{}.dat'.format(self.label))

        return new_lmp_data

    def write_trajectory(self, new_lmp_data, path):
        """
        Get the structure in the form of LAMMPS trajectory, 
        which is to be used as input for the soq/rdf code (by Jan Michael)
        """
        atoms_df = new_lmp_data.atoms

        # time-step: set constant for all structures. Only matters when we
        # calculate averaged SOQ
        time_step = 1000
        # the filename for soq should be of the following format
        filename = path + 'dump.{:09d}.txt'.format(time_step)
        n_atoms = len(atoms_df.x)
        bounds = new_lmp_data.box.bounds
        bounds_lines = [str(bounds[i][0]) + ' ' + str(bounds[i][1]) + '\n' \
                                                    for i in range(3)]
        lines = ['ITEM: TIMESTEP\n', str(time_step) + '\n',
                 'ITEM: NUMBER OF ATOMS\n', str(n_atoms) + '\n',
                 'ITEM: BOX BOUNDS pp pp pp\n'] + bounds_lines + \
                 ['ITEM: ATOMS id type x y z \n']

        with open(filename, 'w') as f:
            f.writelines(lines)
            for i in range(n_atoms):
                index = atoms_df.index[i]
                line = '{0} {1} {2:.6f} {3:.6f} {4:.6f}\n'.format(index,
                                            atoms_df['type'].loc[index],
                                            atoms_df['x'].loc[index],
                                            atoms_df['y'].loc[index],
                                            atoms_df['z'].loc[index])
                f.write(line)
                
    def write_poscar(self, new_lmp_data, filename):
        """
        Get the new structure in the form of POSCAR
        """
        new_lmp_data.structure.to(filename=filename)

    def get_soq(self, data_file):
        """
        Use the soq_mpi_fftv3 to get the SOQ from the data_file
        """
        pass

    def add_polymer_monomers(self, astr):
        """
        If possible, add polymer monomers around NP cavity to increase the
        atom density
        """
