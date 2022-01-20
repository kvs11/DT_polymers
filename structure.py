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
from pymatgen.core.lattice import Lattice

import random
import numpy as np
import pandas as pd
from math import pi
from numpy.random import uniform as unif

from concurrent.futures import ProcessPoolExecutor

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
        self.atom_ids_1 = atoms_df_1.index.to_list()
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

        while NPs_added < n_NP - 1 and new_loc_tries < n_NP+100:
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

    def move_center_to_origin(self, atoms_df, a):
        """
        For a given atoms_df with cartesian coordinates and from a
        lattice of form [0, a], [0, a], [0, a]], get the cartesian
        coordinates for a lattice box centered at origin (0, 0, 0).
        """
        # move center to origin
        trans_vector = np.array([0, 0, 0]) - \
                                np.array([a/2, a/2, a/2])

        # add the trans_vector to all atoms coords
        current_coords = np.array([atoms_df.x.to_list(),
                                   atoms_df.y.to_list(),
                                   atoms_df.z.to_list()]).T

        new_coords = current_coords + trans_vector
        new_atoms_df = atoms_df.copy()
        new_atoms_df["x"] = new_coords[:, 0]
        new_atoms_df["y"] = new_coords[:, 1]
        new_atoms_df["z"] = new_coords[:, 2]

        return new_atoms_df

    def create_PNC(self, poly_atoms_df, NP_atoms_df, topo):
        """
        """
        a = self.latt.a
        new_lmp_box = LammpsBox([[-a/2, a/2], [-a/2, a/2],
                                 [-a/2, a/2]])
        # Note: masses, force_field, atom_style same as that of lmp_data
        pnc_atoms_df = pd.concat([poly_atoms_df, NP_atoms_df])

        pnc_atoms_df = self.move_center_to_origin(pnc_atoms_df, a)
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

    def write_trajectory(self, new_lmp_data, filename):
        """
        Get the structure in the form of LAMMPS trajectory,
        which is to be used as input for the soq/rdf code (by Jan Michael)
        """
        atoms_df = new_lmp_data.atoms

        # time-step: set constant for all structures. Only matters when we
        # calculate averaged SOQ
        # TODO: using hard coded 10000 as timestep, change if needed
        time_step = 10000

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
                # NOTE: use i instead of index to be compatible
                # input for the soq code
                line = '{0} {1} {2:.6f} {3:.6f} {4:.6f}\n'.format(i+1,
                                            atoms_df['type'].loc[index],
                                            atoms_df['x'].loc[index],
                                            atoms_df['y'].loc[index],
                                            atoms_df['z'].loc[index])
                f.write(line)

    def make_type2_trajectory(self, filename, replace=True):
        """
        Modifies a dump file with only type 2 atoms labelled similarly. This is
        to make sure that the S(q) code works properly on different random PNCs
        as opposed to same PNC different trajectories during LAMMPS steps.

        Args:

        filename (str): the filename of the dump file
        replace (bool): state whether to replace the provided filename or write
                        to a new filename
        """
        # get the label and use it as timestep in trajectory
        new_ind = int(int(filename.split('.')[1].split('.')[0]))

        with open(filename) as f:
            lines = f.readlines()

        top_lines = lines[:9]
        # set label as timestep
        top_lines[1] = '{} \n'.format(new_ind)
        # get all type 2 lines
        type_2s = [i for i in lines if " 2 " in i]
        # set no. of total atoms as length of type 2 atoms (or NPs)
        top_lines[3] = '{} \n'.format(len(type_2s))
        new_type_2s = []
        for i, line in enumerate(type_2s):
            x = np.array(line.split())
            x[0] = str(i+1)
            l = "{} {} {} {} {} \n".format(x[0], x[1], x[2], x[3], x[4])
            new_type_2s.append(l)

        if not replace: # write to a new_filename
            filename = 'new_dump.{:09d}.txt'.format(new_ind)
        with open(filename , 'w') as f:
            f.writelines(top_lines)
            f.writelines(new_type_2s)

    def write_poscar(self, new_lmp_data, filename):
        """
        Get the new structure in the form of POSCAR
        """
        new_lmp_data.structure.to(filename=filename)

    def get_soq(self, trajectory):
        """
        Use the soq_mpi_fftv3 to get the SOQ from the data_file
        """
        pass

    def add_polymer_monomers(self, astr):
        """
        If possible, add polymer monomers around NP cavity to increase the
        atom density
        """


class nanoparticles_box(object):

    def __init__(self, np_box_params):
        """
        """
        self.mean_dia_NP
        self.sigma_dia_NP
        self.min_dia_NP
        self.max_dia_NP
        self.min_gap_NP
        self.vol_frac
        pass

    def generate_soq(self, path_to_pnc_dumps, path_to_insoq):
        """
        one soq is made from all the dump files present in the path using the
        in.soq file provided
        """
        pass

    def get_box_latt(n, mean_dia_NP, vol_frac):
        """
        """
        # get the size of the box
        # volume of NPs
        V_nps = n * 4/3 * pi * (mean_np_dia/2)**3
        V_box = V_nps / vol_frac
        box_len = V_box ** (1/3)

        # make lattice
        box_latt = Lattice([[box_len, 0, 0],
                            [0, box_len, 0],
                            [0, 0, box_len]])
        return box_latt

    def get_dia_NP(self):
        """
        """
        found = False
        while not found:
            d = np.random.normal(loc=self.mean_dia_NP, scale=self.sigma_dia_NP)
            if self.min_dia_NP < d < self.max_dia_NP:
                found = True
        return d

    def create_n_pnc_dumps(self, n_NPs, mean_NP_dia, box_latt):
        """
        """
        random_coords = []
        dia_NPs = []
        new_loc_tries = 0
        NPs_added = 0

        while NPs_added < n_NPs - 1 and new_loc_tries < n_NPs+100:
            new_loc_tries += 1
            if len(random_coords) == 0:
                new_fracs = [unif(0, 1), unif(0, 1), unif(0, 1)]
                dia_NP = self.get_dia_NP()
                dia_NPs.append(dia_NP)
                random_coords.append(new_fracs) # frac coords added
                continue
            # Starting from second random coords, check min_dist_NP constraint
            # considering the periodic boundary condition (pbc)
            added_new_fracs = False
            num_tries = 0
            while not added_new_fracs and num_tries < 500:
                num_tries += 1
                # Get the next new_fracs
                new_fracs = [unif(0, 1), unif(0, 1), unif(0, 1)]
                new_carts = box_latt.get_cartesian_coords(new_fracs)
                dia_NP = self.get_dia_NP()
                # Get points within a sphere of max dist for second point
                max_dist = dia_NP/2 + self.max_dia_NP/2 + self.min_gap_NP
                coords_in_new_sphere = box_latt.get_points_in_sphere_py(
                                        random_coords, new_carts, max_dist)

                # check dist individually with each NP within the max_dist
                dist_check = 0
                for each_NP in coords_in_new_sphere:
                    index = each_NP[0]
                    dist = each_NP[1]
                    dia_NP2 = dia_NPs[index]
                    if not dist > dia_NP/2 + dia_NP2/2 + self.min_gap_NP:
                        dist_check += 1
                if dist_check == 0:
                    dia_NPs.append(dia_NP)
                    random_coords.append(new_fracs)
                    added_new_fracs = True
                    NPs_added += 1

    def write_trajectory(self, label, box_latt, cart_coords, filename):
        """
        Get the structure in the form of LAMMPS trajectory,
        which is to be used as input for the soq/rdf code (by Jan Michael)
        """
        # TODO: using hard coded 10000 as timestep, change if needed
        time_step = label * 1000

        if len(cart_coords) != self.n_NPs:
            print ('Warning: Required no. of NPs locs are not obtained')
        n_atoms = len(cart_coords)

        box_half_len = box_latt.a / 2
        bounds_lines = ["-{0} {0}\n-{0} {0}\n-{0} {0}\n".format(box_half_len)]

        # shift all cart coords by box_half_len to match the box bounds
        shifted_cart_coords = cart_coords - box_half_len

        lines = ['ITEM: TIMESTEP\n', str(time_step) + '\n',
                 'ITEM: NUMBER OF ATOMS\n', str(n_atoms) + '\n',
                 'ITEM: BOX BOUNDS pp pp pp\n'] + bounds_lines + \
                 ['ITEM: ATOMS id type x y z \n']

        with open(filename, 'w') as f:
            f.writelines(lines)
            for i, cc in enumerate(cart_coords):
                # NOTE: use i instead of index to be compatible
                line = '{0} 2 {1:.6f} {2:.6f} {3:.6f}\n'.format(i+1,
                                                        cc[0], cc[1], cc[2])
                f.write(line)
