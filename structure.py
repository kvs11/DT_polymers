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

from pymatgen import Structure, Lattice, Element, Composition
from pymatgen.io.lammps.data import LammpsData

import random, copy, time
import numpy as np
from numpy.random import uniform as unif


class polymer(object):
    """
    This class contains all the attributes of a polymer model. Creating random polymer models, or performing GA/mutation operations on a model

    It also reads the intial polymer cell and stores it's supercell. Later, using this cell, multiple models can be generated.

    The purpose of this class in big picture is to make structures which would then be evaluated for S(Q). i.e., velocities and topology (bonds & angles) is not important for this part.
    """
    def __init__(self, polymer_parameters):
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
                                'of {} Å'.format(min_dist_NP))
        else:
            self.min_dist_NP = polymer_parameters['min_dist_NP']

        # Any other parameters needed should be saved here from input parameters


    def get_nNP_and_dia():
        """
        With the given masses of NPs and their mass_fraction in the PNC, get
        the number of NPs needed and their diameter.

        Q: If we vary the size of the NPs in our distribution, how do we
        provide respective mass to each NP and then run LAMMPS?
        """
        pass

    def get_NP_locs(self):
        """
        Get the fractional coordinaees of nanoparticles (NPs) within a box.
        The NPs among themselves should obey a minimum distance constraint.
        The size of each NP should be considered.

        Args:

        n_NP (int): Number of nanp particles need to be added to polymer matrix

        min_dist_NP (float): The minimum distance between two NPs. Defaults to the (2 * dia_NP + 1) Å

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

        return random_coords


    def get_NP_sphere():
        """
        Get the radius of a NP sphere

        NOTE: If we cannot vary the mass of the NPs according to their size
        while doing LAMMPS, then we should not vary size of the NPs.
        """
        pass

    def place_NPs_in_polymer_matrix(self, random_coords):
        """
        Get all the polymer atoms in the NP sphere including the border atoms
        out of sphere with chains within the sphere

        Algorithm:
        1. Provide all the atoms in the supercell (all_fracs) and the center of the NP sphere (center) and radius of the NP sphere (dia_NP/2) to get all atoms in the sphere.

        2. Find all molecule systems with atoms linked within the sphere

        3. Add the indices of these atoms with the atoms within the sphere

        4. Return the indices

        5. Remove all atoms with obtained indices

        6. Add a new atom with "center" as the coordinates and a new species
        """
        atoms_df = self.lmp_data.atoms
        astr = self.lmp_data.structure

        all_fracs = astr.frac_coords
        latt = astr.lattice
        copy_astr = copy.deepcopy(astr)

        atom_locs_to_remove = []
        for center_of_NP in random_coords:
            center_of_NP = latt.get_cartesian_coords(center_of_NP)
            atoms_in_sphere =  latt.get_points_in_sphere(all_fracs,
                                                center_of_NP, self.dia_NP/2)
            # The position/loc of atoms in sphere
            atom_ilocs_in_sphere = [i[2] for i in atoms_in_sphere]

            # get an array of molecule IDs for each atom ID
            mol_ids_in_sphere = np.unique([atoms_df["molecule-ID"].iloc[i] \
                                        for i in atom_ilocs_in_sphere])

            # get all atoms with the corresponding molecule IDs
            for i in mol_ids_in_sphere:
                ith_mol_atoms = atoms_df[atoms_df[
                                    "molecule-ID"]==i].index.to_list()
                ith_mol_atom_locs = [atoms_df.index.get_loc(i) \
                                        for i in ith_mol_atoms]
                atom_locs_to_remove += ith_mol_atom_locs

        atom_locs_to_remove = np.unique(atom_locs_to_remove)
        # remove the atoms for to make all NP cavities
        copy_astr.remove_sites(atom_locs_to_remove)

        # add random coords as a second species with random velocities
        for coords in random_coords:
            props = {'charge': 0.0,
                     'velocities': [unif(-1, 1), unif(-1, 1), unif(-1, 1)]}
            copy_astr.append('Li', coords,
                             coords_are_cartesian=False, properties=props)

        return copy_astr, atom_locs_to_remove


    def create_PNC():
        """
        Make a new polymer nano composite (PNC) with beads and sphere
        """
        pass
