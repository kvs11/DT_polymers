from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import \
    RotationTransformation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import os, random, glob
import copy
import gc

import pandas as pd
import numpy as np
from math import pi
from numpy.random import uniform as unif
from numpy.random import multivariate_normal

from scipy.signal import savgol_filter


class nanoparticles_box(object):

    def __init__(self, np_box_params):
        """
        """
        # Assume default values for all params
        self.mean_dia_NP = np_box_params['mean_dia_NP']
        self.sigma_dia_NP = np_box_params['sigma_dia_NP']
        self.min_dia_NP = np_box_params['min_dia_NP']
        self.max_dia_NP = np_box_params['max_dia_NP']
        self.min_gap_NP = np_box_params['min_gap_NP']
        self.vol_frac = np_box_params['vol_frac']
        self.min_gap_NP_in_cluster = np_box_params['min_gap_NP_in_cluster']
        self.num_clusters = np_box_params['num_clusters']
        self.num_NPs_per_cluster = np_box_params['num_NPs_per_cluster']
        self.cluster_shape = np_box_params['cluster_shape']

        self.min_n_NPs = None
        self.max_n_NPs = None

        self.n_NPs = None
        if 'n_NPs' in np_box_params:
            self.n_NPs = np_box_params['n_NPs']
        self.box_len = None
        if 'box_len' in np_box_params:
            self.box_len = np_box_params['box_len']

        self.box_latt = self.get_box_latt()

    def get_dia_NP(self):
        """
        Function to get the diameter randomly from a normal distribution with 
        user specified mean and sigma.
        """
        found = False
        while not found:
            d = np.random.normal(loc=self.mean_dia_NP, scale=self.sigma_dia_NP)
            if self.min_dia_NP < d < self.max_dia_NP:
                found = True
        return d

    def get_box_latt(self):
        """
        Function to get a box shaped lattice object considering user-provided 
        volume fraction or box_len if available. 
        """
        if self.n_NPs is not None:
            # volume of NPs
            V_nps = self.n_NPs * 4/3 * pi * (self.mean_dia_NP/2)**3
            # get volume of box
            V_box = V_nps / self.vol_frac
            box_len = V_box ** (1/3)
            if self.box_len is None:
                self.box_len = box_len
                print ('For {} NPs and volume fraction {}, box length should '\
                       'be {}'.format(self.n_NPs, self.vol_frac, box_len))
            else:
                if not box_len - self.box_len < 0.1:
                    print ('Warning: Given box_len does not satisfy given'
                            ' vol_frac')
        else:
            V_nps = self.box_len**3 * self.vol_frac
            n_NPs = int(V_nps / (4/3 * pi * (self.mean_dia_NP/2)**3))
            self.n_NPs = n_NPs
            print ('No. of NPs for box len {} and volume fraction {} '\
                   'should be: {}'.format(self.box_len, self.vol_frac, n_NPs))

        # make lattice
        box_latt = Lattice([[self.box_len, 0, 0],
                            [0, self.box_len, 0],
                            [0, 0, self.box_len]])
        return box_latt

    def get_rand_frac_coords(self, curr_NP_coords=None, coords_are_cartesian=True, curr_dia_NPs=None):
        """

        """
        random_coords = []
        dia_NPs = []
        NPs_needed = self.n_NPs
        NPs_added = 0
        # Check if any coords already provided. Then, add them to the list and search for the difference
        if curr_NP_coords is not None:
            random_coords = list(curr_NP_coords)
            if coords_are_cartesian:
                random_coords = list(self.box_latt.get_fractional_coords(
                                                            curr_NP_coords))
            if curr_dia_NPs is None:
                dia_NPs = [self.mean_dia_NP for i in range(len(random_coords))]
            else:
                dia_NPs = list(curr_dia_NPs)
            NPs_added = len(random_coords)
            NPs_needed = self.n_NPs - NPs_added


        new_loc_tries = 0
        # Do finite attepts to find a new coordinate that is not too close with existing set of coordinates.
        while NPs_added < NPs_needed - 1 and new_loc_tries < NPs_needed+100:
            new_loc_tries += 1
            if len(random_coords) == 0:
                # Sample the first coordinate 
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
                # Sample a random coordiante from entire box
                new_fracs = [unif(0, 1), unif(0, 1), unif(0, 1)]
                new_carts = self.box_latt.get_cartesian_coords(new_fracs)
                dia_NP = self.get_dia_NP()
                # Get points within a sphere of max dist for second point
                max_dist = dia_NP/2 + self.max_dia_NP/2 + self.min_gap_NP
                coords_in_new_sphere = self.box_latt.get_points_in_sphere_py(
                                        random_coords, new_carts, max_dist)

                # check dist individually with each NP within the max_dist
                dist_check = 0
                for each_NP in coords_in_new_sphere:
                    dist = each_NP[1]
                    dia_NP2 = dia_NPs[each_NP[2]]
                    if not dist > dia_NP/2 + dia_NP2/2 + self.min_gap_NP:
                        dist_check += 1
                        break
                if dist_check == 0:
                    dia_NPs.append(dia_NP)
                    random_coords.append(new_fracs)
                    added_new_fracs = True
                    NPs_added += 1

        return random_coords, dia_NPs

    def write_trajectory(self, label, astr, filename):
        """
        Get the structure in the form of LAMMPS trajectory,
        which is to be used as input for the soq/rdf code (by Jan Michael)
        """
        # TODO: using hard coded 10000 as timestep, change if needed
        time_step = label * 1000

        cart_coords = astr.cart_coords
        box_half_len = astr.lattice.a / 2

        n_atoms = len(cart_coords)
        bounds_lines = ["-{0} {0}\n-{0} {0}\n-{0} {0}\n".format(box_half_len)]

        # shift all cart coords by box_half_len to match the box bounds
        shifted_cart_coords = cart_coords - box_half_len

        lines = ['ITEM: TIMESTEP\n', str(time_step) + '\n',
                 'ITEM: NUMBER OF ATOMS\n', str(n_atoms) + '\n',
                 'ITEM: BOX BOUNDS pp pp pp\n'] + bounds_lines + \
                 ['ITEM: ATOMS id type x y z \n']

        with open(filename, 'w') as f:
            f.writelines(lines)

            for i, cc in enumerate(shifted_cart_coords):
                # NOTE: use i instead of index to be compatible
                line = '{0} 2 {1:.6f} {2:.6f} {3:.6f}\n'.format(i+1,
                                                        cc[0], cc[1], cc[2])
                f.write(line)

    # Get 100 random PNCs
    def get_rand_pnc_traj(self, label, rand_seed, write_trajectory=True,
                          return_structure=True):
        """
        """
        np.random.seed(rand_seed)
        rand_fracs, dia_NPs = self.get_rand_frac_coords()
        cart_coords = self.box_latt.get_cartesian_coords(rand_fracs)

        if write_trajectory:
            traj_filename = 'dump.{:09d}.txt'.format(label * 1000)
            self.write_trajectory(label, cart_coords, traj_filename)

        # TODO: Remove this bool and make return structure default
        if return_structure:
            sps = ['Li' for i in range(len(cart_coords))]
            struct = Structure(self.box_latt, sps, cart_coords,
                               coords_are_cartesian=True)
            return dia_NPs, struct
        else:
            return dia_NPs

    def get_NP_coords_cluster(self, cluster_dia=None, shuffle_center=False):
        """
        Given maximum allowed diamter of a cluster, this function adds random
        coordinates in a chain like fashion connected to the previous added
        atom which satisfies distance constraints with other atoms present.

        Returns a list of cartesian coordinates

        Args:

        num_NPs (int) - number of atoms needed in the structure

        cluster_dia (float) - maximum diameter of the cluster
        """
        # start from origin
        old_NP_point = np.array([0, 0, 0])

        # Add first point
        rand_cart_coords = []
        rand_cart_coords.append(old_NP_point)
        coords_added = 1
        old_NP_dia = self.get_dia_NP()
        dia_NPs = []
        dia_NPs.append(old_NP_dia)

        new_NP_point_attempt = 0
        while coords_added < self.num_NPs_per_cluster:
            new_NP_dia = self.get_dia_NP()
            dist_NP = old_NP_dia/2 + new_NP_dia/2 + \
                                self.min_gap_NP_in_cluster
            new_NP_point = self.get_point_on_sphere(dist_NP)

            # returns None if the algo cannot add a new point in 500 attempts
            # if the cluster_diameter is too small, this algo hangs trying to
            # add new point
            new_NP_point_attempt += 1
            if new_NP_point_attempt > 1000:
                return None, None

            # translate the point near the old_NP_point
            new_NP_point = new_NP_point + old_NP_point

            # check if the translated point is within cluster diamter box
            if cluster_dia is not None:
                if not np.linalg.norm(new_NP_point) < cluster_dia/2:
                    continue

            # Get points within a sphere of max dist for second point
            max_dist = new_NP_dia/2 + self.max_dia_NP/2 + self.min_gap_NP
            rand_frac_coords = self.box_latt.get_fractional_coords(
                                                        rand_cart_coords)
            coords_in_new_sphere = self.box_latt.get_points_in_sphere_py(
                                    rand_frac_coords, new_NP_point, max_dist)

            # check dist individually with each NP within the max_dist
            dist_check = 0
            for each_NP in coords_in_new_sphere:
                dist = each_NP[1]
                dia_NP2 = dia_NPs[each_NP[2]]
                if dist < new_NP_dia/2 + dia_NP2/2 + \
                                    self.min_gap_NP_in_cluster - 0.1:
                    dist_check += 1
                    break
            if dist_check != 0:
                continue

            # add the new_NP_point and reset the no. of attempts
            dia_NPs.append(new_NP_dia)
            rand_cart_coords.append(new_NP_point)
            new_NP_point_attempt = 0
            old_NP_point = new_NP_point
            if shuffle_center is True:
                old_NP_point = rand_cart_coords[np.random.randint(
                                                    len(rand_cart_coords))]
            coords_added += 1

        if len(rand_cart_coords) < self.num_NPs_per_cluster:
            return None, None
        # coords are cartesian
        return np.array(rand_cart_coords), dia_NPs

    def get_point_on_sphere(self, r):
        """
        Get a random point on a sphere of radius r

        Args:

        r (float) - radius of the sphere
        """

        # get random point (x, y, z) using normal distribution
        point = np.random.randn(3)
        # normalize the point
        point_mag = np.linalg.norm(point)
        point = point / point_mag
        # multiply by radius
        point = point * r

        return point

    def join_n_clusters(self, set_cluster_dia=False, shuffle_center=False):
        """
        """
        num_clusters = self.num_clusters
        if num_clusters > 5:
            # TODO
            print ('num_clusters should be 5 or less.')
        cluster_dia = None
        if set_cluster_dia is True:
            cluster_dia = int(num_clusters**(1/3)*2*2.5)

        all_clusters = []
        all_dia_NPs = []
        while len(all_clusters) < num_clusters:
            clus_carts, dia_NPs = self.get_NP_coords_cluster(
                                            cluster_dia=cluster_dia,
                                            shuffle_center=shuffle_center)
            if clus_carts is not None:
                all_clusters.append(clus_carts)
                all_dia_NPs += dia_NPs

        #### TEMP
        a = self.box_latt.a
        if num_clusters == 1:
            clus_carts = all_clusters[0] + np.array([a/2, a/2, a/2])
            all_clus_carts = clus_carts
        if num_clusters == 2:
            clus_carts = all_clusters[0] + np.array([a/4, a/4, a/4])
            all_clus_carts = clus_carts
            clus_carts = all_clusters[1] + np.array([3*a/4, 3*a/4, 3*a/4])
            all_clus_carts = np.concatenate((all_clus_carts, clus_carts))
        if num_clusters >= 3:
            clus_carts = all_clusters[0] + np.array([a/2, a/2, 0])
            all_clus_carts = clus_carts
            clus_carts = all_clusters[1] + np.array([0, a/2, a/2])
            all_clus_carts = np.concatenate((all_clus_carts, clus_carts))
            clus_carts = all_clusters[2] + np.array([a/2, 0, a/2])
            all_clus_carts = np.concatenate((all_clus_carts, clus_carts))
        if num_clusters >= 4:
            clus_carts = all_clusters[3] + np.array([a/2, a/2, a/2])
            all_clus_carts = np.concatenate((all_clus_carts, clus_carts))
        if num_clusters == 5:
            clus_carts = all_clusters[4]
            all_clus_carts = np.concatenate((all_clus_carts, clus_carts))

        return all_clus_carts, all_dia_NPs

    def get_clustered_pnc_traj(self, label, rand_seed):
        """
        """
        np.random.seed(rand_seed)
        cluster_shape = self.cluster_shape
        if cluster_shape == 'combination':
            cluster_shape = np.random.choice(['chain', 'blob', 'tentacle'])
        if cluster_shape == 'chain':
            set_cluster_dia, shuffle_center = False, False
        elif cluster_shape == 'blob':
            set_cluster_dia, shuffle_center = True, True
        elif cluster_shape == 'tentacle':
            set_cluster_dia, shuffle_center = False, True

        all_clus_carts, all_dia_NPs = self.join_n_clusters(
                                        set_cluster_dia, shuffle_center)
        rand_fracs, dia_NPs = self.get_rand_frac_coords(
                                        curr_NP_coords=all_clus_carts,
                                        coords_are_cartesian=True,
                                        curr_dia_NPs=all_dia_NPs)
        cart_coords = self.box_latt.get_cartesian_coords(rand_fracs)
        traj_filename = 'dump.{:09d}.txt'.format(label * 1000)
        self.write_trajectory(label, cart_coords, traj_filename)
        #"Done {}".format(label)
        return dia_NPs

    def random_translation(self, astr):
        """
        Translates all "atoms" for a random length in x, y and z directions
        """
        cart_coords = astr.cart_coords
        abc = np.array(astr.lattice.abc)
        dxdydz = np.random.uniform(size=3) * abc
        new_cart_coords = cart_coords + dxdydz

        new_astr = Structure(astr.lattice, astr.species, new_cart_coords,
                             coords_are_cartesian=True)
        spg_astr = SpacegroupAnalyzer(new_astr)
        new_astr = spg_astr.get_refined_structure()

        return new_astr

    def random_rotation(self, astr):
        """
        Rotates the "astr" by either [90, 180, 270] degrees along each of x, y
        and z directions one after another
        """
        temp_astr = copy.deepcopy(astr)
        dxdydz = np.random.choice([0, 90, 180, 270], size=3)
        hkls = [[1, 0, 0], [0, 1, 0], [0, 0 ,1]]

        for hkl, angle in zip(hkls, dxdydz):
            rotate = RotationTransformation(hkl, angle)
            temp_astr = rotate.apply_transformation(temp_astr)

        new_latt = temp_astr.lattice
        new_astr = Structure(new_latt, astr.species, astr.cart_coords,
                             coords_are_cartesian=True)
        spg_astr = SpacegroupAnalyzer(new_astr)
        new_astr = spg_astr.get_refined_structure()

        return new_astr

    def _get_abc_sets(self, n):
        """
        """
        if n > 10:
            print ("10x10x10 is too big for a supercell size in 3D. Using 5..")

        abc_sets = []
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    abc_sets.append([x, y, z])
        abc_sets.remove([0, 0, 0])

        return abc_sets

    def increase_pnc_size(self, init_astr, size=2, alter_cells=True):
        """
        For size 2, makes 2x2x2 = 8x the initial structure
        For size 3, makes 3x3x3 = 27x the initial structure

        Example: For size=2,
        get 7 transformed sets of cart_coords
        translate each set into respective locations i.e.,
        {a, 0, 0 ; 0, b, 0 ; 0, 0, c ; a, b, 0; 0, b, c ; a, 0, c; a, b, c}
        """
        init_carts = init_astr.cart_coords
        init_min_xyz = init_carts.min(axis=0)
        abc = np.array(init_astr.lattice.abc)
        abc_sets = self._get_abc_sets(size)

        size_n_carts = list(init_carts)
        for new_abc in abc_sets:
            if alter_cells:
                translated_astr = self.random_translation(init_astr)
                rotated_astr = self.random_rotation(translated_astr)
                new_carts = rotated_astr.cart_coords
            else:
                new_carts = init_carts.copy()
            new_min_xyz = init_min_xyz + abc * new_abc
            curr_min_xyz = new_carts.min(axis=0)
            translate_xyz = new_min_xyz - curr_min_xyz
            new_carts = new_carts + translate_xyz
            size_n_carts += list(new_carts)

        size_n_latt = init_astr.lattice.matrix * size
        size_n_sps = ['Li' for i in range(len(size_n_carts))]
        size_n_astr = Structure(size_n_latt, size_n_sps, size_n_carts,
                                coords_are_cartesian=True)

        size_n_astr.sort()

        return size_n_astr

###########################################################################

import os
import time
import subprocess as sp
import numpy as np
import random
from math import pi
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize
from scipy.optimize import minimize
#from NPs_box import nanoparticles_box as np_box

class register_id(object):
    def __init__(self):
        self.label = 0
    def create_id(self):
        self.label += 1
        return self.label

class candidate(object):
    def __init__(self, reg_id, param_set, chi_stat=999999):
        self.label = reg_id.create_id()
        self.params = param_set
        self.chi_stat = chi_stat
        self.selection_prob = 0
        self.max_times_as_parent = 20
        self.times_chosen_as_parent = 0

def get_xy(f):
    with open(f) as ff:
        lines = ff.readlines()
        x = [float(i.split()[0]) for i in lines]
        y = [float(i.split()[1]) for i in lines]
        return np.array([x[1:], y[1:]])

def change_dump_ids():
    """
    change atom ids in dump file to be compatible with the soq code
    """
    # read the dump file
    with open('dump.000000000.txt') as f:
        lines = f.readlines()
    # for each dump file, readlines and write new lines with updated atom ID
    all_dumps = [i for i in os.listdir() if i.startswith('dump')]
    for dump_file in all_dumps:
        with open (dump_file) as f:
            lines = f.readlines()
            atom_lines = lines[9:]
            new_lines = []
        for i, l in enumerate(atom_lines):
            spt = l.split()
            nl = '{0} {1} {2} {3} {4} \n'.format(i+1, spt[1],
 spt[2],
                                                      spt[3], spt[4])
            new_lines.append(nl)
            new_lines = lines[:9] + new_lines
            with open(dump_file, 'w') as f:
                f.writelines(new_lines)

def get_n_NP(dia_NP, vol_box):
    """
    Keep the volume fraction between 0.55 and 0.75
    0.55* vol_box * 3/4pi <= r^3 * n_NP <= 0.75* vol_box * 3/4pi
    Parameters
    ----------

    dia_NP : float
        Diameter of NP
    vol_box : float
        Volume of the entire PNC box


    Returns
    -------
    (int)
    number of NPs
    """
    vol_NP = 4/3 * pi * (dia_NP/2)**3
    min_n_NP = int(0.07*vol_box/vol_NP)
    max_n_NP = int(0.11*vol_box/vol_NP)
    n_NP = random.randint(min_n_NP, max_n_NP)
    return n_NP

# get simulated soq from the 50 PNCs
def get_sim_soq(pnc_params, candidate_dir):
    """
    """
    # Make one simulated SOQ with all the scells' trajectory files
    soq_input_file = pnc_params['in_soq_path']
    scells_for_one_soq = pnc_params['scells_for_one_soq']
    cmd = pnc_params['soq_exec_cmd']

    with open(soq_input_file) as f:
        lines = f.readlines()
    lines[1] = 'endStep {}\n'.format(scells_for_one_soq*1000)
    lines[3] = 'initStep 1000\n'
    to_file = candidate_dir + '/in.soq'
    with open(to_file, 'w') as f:
        f.writelines(lines)

    with open(to_file) as inp:
        sp.Popen(cmd.split(), stdout=sp.PIPE, stdin=inp).communicate()

def get_init_rescaling(pnc_params, candidate_dir):
    """
    Function to get initial rescaling factor. The one with maxima matching..
    """
    # get simulated x and y from the candidate soq.txt file
    sim_soqf = candidate_dir + '/soq.txt'
    sim_x, sim_y = get_xy(sim_soqf)

    # get experimental x and y from the candidate soq.txt file
    exp_soq_file = pnc_params['exp_soq_path']
    exp_q, exp_soq = get_xy(exp_soq_file)
    exp_max_ind = np.argmax(exp_soq)
    exp_q_val_max = exp_q[exp_max_ind]

    sim_max_ind = np.argmax(sim_y)
    sim_q_val_max = sim_x[sim_max_ind]

    #print (exp_q_val_max, sim_q_val_max)

    # new_scaling_factor to have same q for their first peaks
    rescaling_factor = exp_q_val_max / sim_q_val_max
    #print (new_scaling_factor)

    # Convert q(sigma-1) --> q(A-1)
    # new_scaling_factor = 0.0201406373292868
    sim_x = sim_x * rescaling_factor
    return rescaling_factor

# define the function to be looped
def get_candidate_soq_residual(cand):
    """
    """
    # get the parameters
    pnc_params = cand.params
    scell_size = pnc_params['scell_size']
    np_box_obj = nanoparticles_box(pnc_params)

    # Make a directory for the candidate -
    # For one candidate SOQ with a label, need scells_for_one_soq PNCs & scells
    soq_label = cand.label
    candidate_dir = pnc_params['main_path'] + '/calcs/{}'.format(soq_label)
    os.mkdir(candidate_dir)
    os.chdir(candidate_dir)
    # write the paramters to the candidate directory
    write_param_set(candidate_dir, pnc_params)

    # Make "scells_for_one_soq" supercells with alterations
    scells_for_one_soq = pnc_params['scells_for_one_soq']
    for i_traj in range(1, scells_for_one_soq+1):
        rand_n = np.random.randint(1000000)
        # NOTE: This step sometimes takes too long. 
        # So, for the first 3 attempts, check if time taken 
        # to make one PNC is more than 100 secs, 
        # then cancel this candidate and return -1
        start_time = time.time()
        _, pnc_astr = np_box_obj.get_rand_pnc_traj(i_traj, rand_n,
                                            write_trajectory=False,
                                            return_structure=True)
        if i_traj < 4 and time.time() - start_time > 300:
            with open('time_taken.txt', 'w') as f:
                f.write('{}'.format(time.time() - start_time))
            return -1
        pnc_scell = np_box_obj.increase_pnc_size(pnc_astr, size=scell_size,
                                                 alter_cells=True)
        traj_filename = candidate_dir + '/dump.{:09d}.txt'.format(i_traj * 1000)
        np_box_obj.write_trajectory(i_traj, pnc_scell, traj_filename)

    # Make one simulated SOQ with all the scells trajectory files
    soq_input_file = pnc_params['in_soq_path']
    with open(soq_input_file) as f:
        lines = f.readlines()
    lines[1] = 'endStep {}\n'.format(scells_for_one_soq*1000)
    lines[3] = 'initStep 1000\n'
    to_file = candidate_dir + '/in.soq'
    with open(to_file, 'w') as f:
        f.writelines(lines)
    cmd = pnc_params['soq_exec_cmd']
    with open(to_file) as inp:
        sp.Popen(cmd.split(), stdout=sp.PIPE, stdin=inp).communicate()

    # Optimize the rescaling factor (rf)
    init_rf = [0.02]
    opt_rf = minimize(get_chi_squared_stat, init_rf,
                                args=(candidate_dir, pnc_params),
                                method='L-BFGS-B',
                                bounds=[(0.011, 0.033)],
                                options={'maxiter': 10000})

    chi_squared_stat = get_chi_squared_stat(opt_rf.x, candidate_dir,
                                            pnc_params, plot=True)

    with open(candidate_dir+'/output.txt', 'w') as f:
        f.write('\nOptimized RF: {}\n'.format(opt_rf.x))
        f.write('Chi-squared stat: {}\n'.format(chi_squared_stat))

    return chi_squared_stat

def get_random_param_set(i_dict):
    """
    """
    fixed_params = i_dict['pnc_fixed_params']
    variable_params = i_dict['pnc_variables']
    variable_bounds = i_dict['bounds']

    random_variable_params = {}
    for key in variable_params:
        random_variable_params[key] = np.random.uniform(
                    variable_bounds[key][0], variable_bounds[key][1])

    if 'num_clusters' not in fixed_params.keys() and \
                'num_NPs_per_cluster' not in fixed_params.keys():
        for key in ['num_clusters', 'num_NPs_per_cluster']:
            random_variable_params[key] = np.random.randint(
                                    variable_bounds[key][0],
                                    variable_bounds[key][1]+1)
        # redo the cluster parameters with a condition
        while random_variable_params['num_clusters'] * \
                random_variable_params['num_NPs_per_cluster'] > 60:
            random_variable_params['num_NPs_per_cluster'] = np.random.randint(
                    variable_bounds['num_NPs_per_cluster'][0],
                    variable_bounds['num_NPs_per_cluster'][1]+1)
    random_param_set = {**fixed_params, **random_variable_params}
    return random_param_set

def update_selection_probs(all_candidates):
    """
    """
    # get chi squared values of all candidates
    chi_squared_values = np.array([cand.chi_stat for cand in all_candidates])
    
    # Custom scaler to avoid outliers influence
    # Same as MinMax scaler but instead of Max we use 75th percentile
    chi2_min = chi_squared_values.min()
    chi2_max = np.percentile(chi_squared_values, 75)
    scaled_chi2_values = (chi_squared_values - chi2_min)/(chi2_max - chi2_min)
    selection_probs = 1 - scaled_chi2_values
    selection_probs[selection_probs < 0] = 0

    # assign selection probs to candidates
    for cand, selection_prob in zip(all_candidates, selection_probs):
        cand.selection_prob = selection_prob

def write_param_set(candidate_dir, candidate_params):
    """
    """
    # write the parameters to the candidate directory
    with open(candidate_dir + '/params.txt', 'w') as f:
        for key, val in candidate_params.items():
            f.write('{} {}\n'.format(key, val))

def get_a_parent(all_candidates, initial_population):
    """
    Returns exactly one parent
    """
    done = False
    num_attempts = 0
    while not done and num_attempts < 100:
        num_attempts += 1
        # randomly choose a parent
        parent = random.choice(all_candidates)
        if parent.times_chosen_as_parent > parent.max_times_as_parent:
            if len(all_candidates) < initial_population:
                print (f"Length of all_candidates: {len(all_candidates)} is low. "\
                        "Wait till new candidates are evaluated.")
                return None
            # remove parent from good_pool and continue
            print (f"Candidate {parent.label} reached maximum times chosen. Removing from all_candidates..")
            all_candidates.remove(parent)
            continue
        if parent.selection_prob:
            if random.random() < parent.selection_prob:
                return parent
    if num_attempts >= 100:
        print (f"get_a_parent could not get viable parent in 100 attempts.")
        return None

def get_two_parents(all_candidates, initial_population):
    cand_1 = get_a_parent(all_candidates, initial_population)
    # Setting a maximum number of attempts to create parents to avoid exhaustively 
    # creating new set of params while generated ones are being evaluated 
    num_attempts = 0
    while num_attempts < 10:
        num_attempts += 1
        cand_2 = get_a_parent(all_candidates, initial_population)
        if cand_1 is None or cand_2 is None:
            print ("get_a_parent returned None!!")
            return None
        if cand_1.label != cand_2.label:
            break
    if num_attempts >= 10:
        print ("Could not find second different parent in 10 attempts.")
        return None
    return cand_1, cand_2

def get_child_param_set(all_candidates, i_dict, mode='avg'):
    """
    """
    fixed_params = i_dict['pnc_fixed_params']
    variable_params = i_dict['pnc_variables']
    initial_population = i_dict['initial_population']

    cand_1, cand_2 = get_two_parents(all_candidates, initial_population)

    # get the new parameters
    new_params = {}
    if mode == 'avg': 
        for key in variable_params.keys():
            new_params[key] = (cand_1.params[key] + cand_2.params[key])/2
        cand_1.times_chosen_as_parent += 1
        cand_2.times_chosen_as_parent += 1
        print (f"New candidate is created by 'avg': {cand_1.label}, {cand_2.label}")

    elif mode == 'crossover':
        keys = list(variable_params.keys())
        np.random.shuffle(keys)
        for i, key in enumerate(keys):
            if i%2 == 0:
                new_params[key] = cand_1.params[key]
            else:
                new_params[key] = cand_2.params[key]
        cand_1.times_chosen_as_parent += 1
        cand_2.times_chosen_as_parent += 1
        print (f"New candidate is created by 'crossover': {cand_1.label}, {cand_2.label}")

    elif mode == 'exploit':
        for key in variable_params.keys():
            variable = cand_1.params[key]
            new_params[key] = np.random.normal(loc=variable, scale=variable*0.2)
        cand_1.times_chosen_as_parent += 1
        print (f"New candidate is created by 'exploit': {cand_1.label}")

    for key in ['num_clusters', 'num_NPs_per_cluster']:
        if key not in fixed_params.keys():
            if not isinstance(new_params[key], int):
                try:
                    new_params[key] = int(new_params[key])
                except:
                    int_param = int(new_params[key])
                    new_params[key] = np.random.choice([int_param, int_param+1])
    if 'num_clusters' not in fixed_params.keys() and \
            'num_NPs_per_cluster' not in fixed_params.keys():
        while new_params['num_clusters'] * new_params['num_NPs_per_cluster'] > 60:
            new_params['num_NPs_per_cluster'] += -1

    return {**fixed_params, **new_params}                         

def get_working_jobs(futures):
    """
    Checks if any jobs in futures is still running and returns number of
    running jobs

    Args:

    futures - list of future objects (concurrent.futures)
    """
    if len(futures) == 0:
        return 0
    else:
        running = 0
        for future in futures:
            if not future.done():
                running += 1

        return running

def update_pool(evald_futures, all_candidates, data_file):
    """
    Calculates the obejctive values for all models and updates pool with
    best models

    Returns updated (evald_futures, pool, models_evald)

    Args:

    evald_futures : (list) list of submitted energy evaluation futures objects

    models_evald : (int) count of number of fully evaluated models

    pool : pool object from selection.py

    select : select object from selection.py

    data_file : path to data_file to write model data

    sim_ids : (bool) True if experimental simulation is used
    """
    # remove all futures with an exception
    rem_inds, process_inds = [], []
    for i, future in enumerate(evald_futures):
        if future.done():
            rem_inds.append(i)
            if not future.exception():
                process_inds.append(i)

    # get all futures which should be processed
    futures_to_process = [evald_futures[i] for i in process_inds]
    # remove all done futures from evald_futures
    evald_futures = [evald_futures[i] for i in range(len(evald_futures))
                     if i not in rem_inds]
    for future in futures_to_process:
        cand = future.result()
        if cand == -1:
            continue
        all_candidates.append(cand)
        print (f"New candidate {cand.label} is added to all_candidates!")
        write_data(cand, data_file)

    return evald_futures, all_candidates

def write_data(cand, file_path):
    """
    """
    with open(file_path, 'a') as f:
        f.write('{} {}\n'.format(cand.label, cand.chi_stat))

def get_chisquare_weighted(observed_values, expected_values, q_array, weights_dict):
    """
    weights_dict = {"q_extrema": [0.033, 0.047, 0.063],
                    "weights": [100, 90, 80], 
                    "q_tol": 0.005}
    """
    q_tol = weights_dict['q_tol']
    q_extrema = weights_dict['q_extrema']
    weights_arr = np.ones(len(observed_values))
    for i, q in enumerate(q_array):
        for j in range(len(q_extrema)):
            if q_extrema[j] - q_tol <= q <= q_extrema[j] + q_tol:
                weights_arr[i] = weights_dict['weights'][j]
                break

    weighted_statistic=0
    for observed, expected, w in zip(observed_values, expected_values, weights_arr):
        weighted_statistic += ((float(observed)-float(expected))**2 *w) / float(expected)

    return weighted_statistic 

def get_q_soq_extrema(exp_q, exp_soq, skip=7):
    """
    
    """
    # First, downsample 1 point for every 10 points
    # This is to avoid noise 
    q = [exp_q[i*skip] for i in range(int(len(exp_q)/skip - 1))]
    soq = [exp_soq[i*skip] for i in range(int(len(exp_q)/skip - 1))]

    # get differential of soq with respect to q
    soq_diff = np.diff(soq)
    # get q values diff
    q_diff = np.diff(q)
    # get slope: divide soq_diff by q_values diff
    slope = soq_diff / q_diff
    # find where slope changes sign
    sign_change_inds = np.where(np.diff(np.sign(slope)))[0]
    sign_change_inds = sign_change_inds + 1
    # return q values at those points
    q_extrema, soq_extrema = [], [] 
    for i in sign_change_inds:
        q_extrema.append(q[i])
        soq_extrema.append(soq[i])

    return q_extrema, soq_extrema

def chisquare(observed_values,expected_values):
    test_statistic=0
    for observed, expected in zip(observed_values, expected_values):
        test_statistic+=(float(observed)-float(expected))**2/float(expected)
    return test_statistic

def get_chi_squared_stat(rescaling_factor, candidate_dir, pnc_params, plot=False):
    """
    """
    qrange = pnc_params['qrange']
    rescaling_factor = rescaling_factor[0]
    # align maxima and rescale
    # experimental SOQ should be only data of the form: 'q    Soq\n'
    exp_soq_file = pnc_params['exp_soq_path']
    exp_q, exp_soq = get_xy(exp_soq_file)

    # get simulated x and y from the candidate soq.txt file
    sim_soqf = candidate_dir + '/soq.txt'
    sim_x, sim_y = get_xy(sim_soqf)
    # rescale sim_x (values of q)
    sim_x = sim_x * rescaling_factor

    # Do spline interpolation on exp Soq & sim Soq

    # First, downsample 1 point for every 10 points for exp soq
    # This is to avoid noise 
    skip = 10
    downd_q = [exp_q[i*skip] for i in range(int(len(exp_q)/skip - 1))]
    downd_soq = [exp_soq[i*skip] for i in range(int(len(exp_q)/skip - 1))]

    # get exp data for interpolation
    exp_min_ind = min(range(len(downd_q)), key=lambda i: abs(downd_q[i]-qrange[0]))
    exp_max_ind = min(range(len(downd_q)), key=lambda i: abs(qrange[1]-downd_q[i]))
    exp_new_x   = downd_q[exp_min_ind-1:exp_max_ind+1]
    exp_new_y   = downd_soq[exp_min_ind-1:exp_max_ind+1]
    # get sim data for interpolation
    sim_min_ind = min(range(len(sim_x)), key=lambda i: abs(sim_x[i]-qrange[0]))
    sim_max_ind = min(range(len(sim_x)), key=lambda i: abs(qrange[1]-sim_x[i]))
    sim_new_x   = sim_x[sim_min_ind-1:sim_max_ind+1]
    sim_new_y   = sim_y[sim_min_ind-1:sim_max_ind+1]
    # get q values within this range and then interpolated exp, sim soq values
    num_pts = 1000
    qvalues = np.linspace(qrange[0], qrange[1], num_pts)
    exp_spline  = InterpolatedUnivariateSpline(exp_new_x, exp_new_y)
    exp_yi = exp_spline(qvalues)
    sim_spline  = InterpolatedUnivariateSpline(sim_new_x, sim_new_y)
    sim_yi = sim_spline(qvalues)

    # observed_values, expected_values, q_array, weights_dict
    weights_dict = {"q_extrema": [],
                    "weights": pnc_params["weights"], 
                    "q_tol": pnc_params["q_tol"]}

    # get q_extrema and soq_extrema
    q_extrema, soq_extrema = get_q_soq_extrema(qvalues, exp_yi)
    weights_dict["q_extrema"] = q_extrema[:len(weights_dict["weights"])]    

    weighted_chisquared = get_chisquare_weighted(sim_yi, exp_yi, qvalues, weights_dict)
    if plot:
        plt.scatter(qvalues, exp_yi, marker='o', facecolors='none', edgecolors='g')
        plt.plot(qvalues, sim_yi, c='orange')
        plt.xlabel(r'q($\sigma^{-1}$)', fontsize=14)
        plt.ylabel('S(q)', fontsize=14)
        plt.text(0.11, 0.1, r'$\chi^{}$ : {:.3f}'.format(2, weighted_chisquared), fontsize=14)
        plt.legend(['Exp. S(q)', 'Sim. S(q)'])
        plt.savefig(candidate_dir + '/soq_plot.png')
        plt.close()

    return weighted_chisquared

def soq_chi2_savgol(rescaling_factor, candidate_dir, pnc_params, plot=False):
    """
    """
    qrange = pnc_params['qrange']
    rescaling_factor = rescaling_factor[0]
    # align maxima and rescale
    # experimental SOQ should be only data of the form: 'q    Soq\n'
    exp_soq_file = pnc_params['exp_soq_path']
    exp_q, exp_soq = get_xy(exp_soq_file)
    # Denoise with Savitzky-Golay filter
    denoised_exp_soq = savgol_filter(exp_soq, 51, 2) # window size of 50 is good for smoothening exp S(q)

    # get simulated x and y from the candidate soq.txt file
    sim_soqf = candidate_dir + '/soq.txt'
    sim_x, sim_y = get_xy(sim_soqf)
    # rescale sim_x (values of q)
    sim_x = sim_x * rescaling_factor

    # Denoise with Savitzky-Golay filter
    denoised_sim_y = savgol_filter(sim_y, 101, 2) # window size 100, polynomial order 2 good for sim soq

    # Perform spline interpolation on denoised experimental SOQ & sim SOQ
    num_pts = 1000
    qvalues = np.linspace(qrange[0], qrange[1], num_pts)
    exp_spline  = InterpolatedUnivariateSpline(exp_q, denoised_exp_soq)
    exp_yhat = exp_spline(qvalues)
    sim_spline  = InterpolatedUnivariateSpline(sim_x, denoised_sim_y)
    sim_yhat = sim_spline(qvalues)

    # Minimize Chi2 to get optimum rescaling_factor
    # observed_values, expected_values, q_array, weights_dict
    weights_dict = {"q_extrema": [],
                    "weights": pnc_params["weights"], 
                    "q_tol": pnc_params["q_tol"]}

    # get q_extrema and soq_extrema
    q_extrema, soq_extrema = get_q_soq_extrema(qvalues, exp_yhat)
    weights_dict["q_extrema"] = q_extrema[:len(weights_dict["weights"])]    

    weighted_chisquared = get_chisquare_weighted(sim_yhat, exp_yhat, qvalues, weights_dict)
    if plot:
        plt.tick_params(labelsize=18)
        ax = plt.gca()
        plt.scatter(qvalues, exp_yhat, marker='o', facecolors='none', edgecolors='g')
        plt.plot(qvalues, sim_yhat, c='orange', linewidth=3)
        plt.xlabel(r'q($\sigma^{-1}$)', fontsize=22)
        plt.ylabel('S(q)', fontsize=22)
        plt.text(0.65, 0.9, r'$\chi^{}$ : {:.2f}'.format(2, weighted_chisquared), fontsize=18, transform=ax.transAxes)
        plt.legend(['Exp. S(q)', 'Sim. S(q)'], fontsize=18, loc=4)
        plt.tight_layout()
        plt.savefig(candidate_dir + '/soq_plot.png')
        plt.close()

    return weighted_chisquared

###############################################################################

def dist(p0, p1):
    """
    Get distance between two points in 3D
    """
    return np.linalg.norm(np.array(p0) - np.array(p1))

def get_point_on_sphere(r):
    """
    Returns a random point on a sphere of radius r

    Args:

    r (float): radius of the sphere
    """

    # get random point (x, y, z) using normal distribution
    point = np.random.randn(3)
    # normalize the point
    point_mag = np.linalg.norm(point)
    point = point / point_mag
    # multiply by radius
    point = point * r

    return point

def get_neighbor_voxels(current_voxel):
    """
    Function to obtain the neighboring indices of current voxel index
    """
    # Consider [xi, yi, zi] as the current voxel
    xi, yi, zi = current_voxel
    neighbor_voxels = []
    for neighbor_x in [xi-1, xi, xi+1]:
        for neighbor_y in [yi-1, yi, yi+1]:
            for neighbor_z in [zi-1, zi, zi+1]:
                neighbor_xyz = [neighbor_x, neighbor_y, neighbor_z]
                neighbor_voxels.append(neighbor_xyz)
    # Remove the voxel itself from its neighbor list
    neighbor_voxels.remove([xi, yi, zi])
    return neighbor_voxels

def perturbation_satisfies_dists(NP_index, astr, dia_NPs, min_gap_NP, new_cart_coords):
    """
    Returns True if the perturbation to a NP coordinates in PNC satisfies dists with neighbors
    """
    curr_dia_NP = dia_NPs[NP_index]
    max_dia_NP = dia_NPs.max()
    all_frac_points = astr.frac_coords
    limiting_dist = max_dia_NP/2 + curr_dia_NP/2 + min_gap_NP

    atoms_nearby = \
        astr.lattice.get_points_in_sphere(all_frac_points, new_cart_coords, limiting_dist)

    # Delete the original coords from atoms nearby
    for i, atom_data in enumerate(atoms_nearby):
        if atom_data[2] == NP_index:
            duplicate_atom_ind = i
            del atoms_nearby[duplicate_atom_ind]
            break

    dists_ok = True
    if len(atoms_nearby) == 0:
        return dists_ok

    dists_nearby = [i[1] for i in atoms_nearby]
    inds_nearby = [i[2] for i in atoms_nearby]

    for i, dist in enumerate(dists_nearby):
        neighbor_dia = dia_NPs[inds_nearby[i]]
        if dist < curr_dia_NP/2 + neighbor_dia/2 + min_gap_NP:
            dists_ok = False
            return dists_ok
    return dists_ok
    
def get_initial_trajectory(rand_seed, np_box_obj, rv_distribution):
    np.random.seed(rand_seed)
    NPs_needed = np_box_obj.n_NPs
    box_len = np_box_obj.box_len
    min_gap_NP = np_box_obj.min_gap_NP

    zones_in_x = zones_in_y = zones_in_z = int(NPs_needed**(1/3))
    # We need to increase zones in x, y and z by 1 to accommodate all NPs_needed
    for i in range(3):
        grid_zones = zones_in_x * zones_in_y * zones_in_z
        if grid_zones >= NPs_needed:
            # confirm zones
            break
        if i == 0:
            zones_in_x += 1
        elif i == 1:
            zones_in_y += 1
        else:
            zones_in_z += 1
            grid_zones = zones_in_x * zones_in_y * zones_in_z
    print (f"Total {grid_zones} zones in box for {NPs_needed} NPs")
    # So, randonly choose to skip some zones [xi_skip, yi_skip, zi_skip]
    skip_diff = grid_zones - NPs_needed
    skip_zones = []
    for i in range(skip_diff):
        done = False
        while not done:
            skip_zone = [np.random.randint(0, zones_in_x), np.random.randint(0, zones_in_y), np.random.randint(0, zones_in_z)]
            if skip_zone not in skip_zones:
                skip_zones.append(skip_zone)
                done = True

    zone_lens = [box_len/zones_in_x, box_len/zones_in_y, box_len/zones_in_z]
    zone_len_x, zone_len_y, zone_len_z = zone_lens
    # Get the cartesian center for the a voxel zone
    zone_center = np.array(zone_lens) / 2

    # Make a numpy array of size nx, ny, nz
    voxel_box = np.zeros((zones_in_x, zones_in_y, zones_in_z, 4))

    rv_distribution = 'uniform' # or 'normal'
    NPs_added = 0
    # Iterate through each voxel zone and get a location for the NP
    for xi in range(zones_in_x):
        for yi in range(zones_in_y):
            for zi in range(zones_in_z):
                if [xi, yi, zi] in skip_zones:
                    continue
                dists_ok = False
                tries = 0
                while dists_ok is False and tries < 200:
                    tries += 1 
                    # Get the diameter of the NP in this zone
                    dia_NP = np_box_obj.get_dia_NP()
                    # Get the mean and sigma for this zone
                    translate_zone = np.array([xi * zone_len_x, yi * zone_len_y, zi * zone_len_z])
                    # Mean is the center of this current voxel zone in cartesian
                    mean_zone = zone_center + translate_zone
                    # Sigma is based on the diameter such that the NP does not cross the zone
                    sigma_zone = (np.array(zone_lens) - dia_NP) / 3 # 3sigma is within the box (99.9%)
                    covar_zone = np.diag(sigma_zone)
                    # Draw random sample from this gaussian
                    if rv_distribution == 'normal':
                        rand_arr = multivariate_normal(mean=mean_zone, cov=covar_zone)
                    elif rv_distribution == 'uniform':
                        xi_min, yi_min, zi_min = mean_zone - zone_center + dia_NP/2
                        xi_max, yi_max, zi_max = mean_zone + zone_center - dia_NP/2
                        rand_arr = np.array([unif(xi_min, xi_max), unif(yi_min, yi_max), unif(zi_min, zi_max)])
                    diff = rand_arr - mean_zone
                    if np.any(abs(diff[:]) > sigma_zone):
                        continue
                    # Optional: Check if NP crossing the zone with NP centered at rand_arr
                    # check distances with neighbor voxels
                    dists_ok = True
                    neighbor_voxels = get_neighbor_voxels([xi, yi, zi])
                    for arr in neighbor_voxels:
                        if len(arr) > 3:
                            if dist(rand_arr, arr[:3]) < min_gap_NP + (dia_NP + arr[3])/2:
                                dists_ok = False
                # Add the rand_arr as NP center to voxel_box
                voxel_box[xi, yi, zi] = np.append(rand_arr, dia_NP)
                NPs_added += 1
    
    return voxel_box, NPs_needed==NPs_added

def get_initial_astr(rand_seed, np_box_obj):
    # Make the initial pnc_astr
    rv_distribution = 'uniform' 
    voxel_box, all_NPs_added = get_initial_trajectory(rand_seed, np_box_obj, rv_distribution)
    if all_NPs_added: print (f"NPs added if same as NPs needed!")
    # No. of zones in voxel box
    grid_zones = voxel_box.shape[0] * voxel_box.shape[1] * voxel_box.shape[2]

    # Create a Structur object for initial trajectory
    latt = Lattice([[np_box_obj.box_len, 0, 0], [0, np_box_obj.box_len, 0], [0, 0, np_box_obj.box_len]])
    cart_coords = voxel_box[:, :, :, :3].reshape(grid_zones, 3)
    # Delete all place holder zero coords from the cart_coords
    cart_coords = np.delete(cart_coords, np.where((cart_coords==np.array([0,0,0])).all(-1))[0], axis=0)
    sps = ['Li' for i in range(len(cart_coords))]
    # Get the dia_NPs
    dia_NPs = voxel_box[:, :, :, 3].reshape(grid_zones)
    dia_NPs = np.delete(dia_NPs, np.where(dia_NPs==0))
    # Structure object
    astr = Structure(latt, sps, cart_coords, coords_are_cartesian=True)
    astr.to(filename='POSCAR_NPbox2', fmt='poscar')
    return dia_NPs, astr

def get_perturbed_trajectory(np_box_obj, dia_NPs, copy_astr, min_perturbation, max_perturbation, check_distances=True):
    cart_coords = copy_astr.cart_coords
    min_gap_NP = np_box_obj.min_gap_NP

    num_perturbed = 0
    for i, one_coords in enumerate(cart_coords):
        replaced = False
        tries = 0
        while not replaced and tries < 100: # try to perturb an an NP in 100 attempts
            tries += 1
            # Max perturbation in Å
            tries_frac = round((tries-5)/100, 1)
            perturbation_range_step = tries_frac * (max_perturbation - min_perturbation)
            high = max_perturbation - perturbation_range_step
            low = high - (max_perturbation - min_perturbation)/10
            jump = unif(low, high)
            perturb = get_point_on_sphere(jump)
            new_cart_coords = one_coords + perturb
            # Check distance and replace with new coords
            if check_distances:
                if perturbation_satisfies_dists(i, copy_astr, dia_NPs, min_gap_NP, new_cart_coords):
                    copy_astr.replace(i, 'Li', new_cart_coords, coords_are_cartesian=True)
                    replaced = True
                    num_perturbed += 1
            else:
                copy_astr.replace(i, 'Li', new_cart_coords, coords_are_cartesian=True)
                replaced = True
                num_perturbed += 1
    return copy_astr,num_perturbed

# define the function to be looped
def get_candidate_soq_residual_V2(cand, pnc_params):
    """
    """
    # get the parameters
    pnc_params = cand.params
    min_perturbation = pnc_params['min_perturbation']
    max_perturbation = pnc_params['max_perturbation']
    check_perturbation_dists = pnc_params['check_perturbation_dists']
    scell_size = pnc_params['scell_size']
    scells_for_one_soq = pnc_params['scells_for_one_soq']
    np_box_obj = nanoparticles_box(pnc_params)

    # Make a directory for the candidate -
    # For one candidate SOQ with a label, need scells_for_one_soq PNCs & scells
    soq_label = cand.label
    candidate_dir = pnc_params['main_path'] + '/calcs/{}'.format(soq_label)
    os.mkdir(candidate_dir)
    os.chdir(candidate_dir)
    # write the paramters to the candidate directory
    write_param_set(candidate_dir, pnc_params)

    # Get the initial trajectory astr
    rand_int = np.random.randint(1000000) # random seed
    dia_NPs, init_astr = get_initial_astr(rand_int, np_box_obj)
    # perturbation begins here
    copy_astr = copy.deepcopy(init_astr)
    for i_traj in range(1, scells_for_one_soq+1):
        copy_astr, num_perturbed = get_perturbed_trajectory(np_box_obj, dia_NPs, copy_astr, min_perturbation, max_perturbation, check_distances=check_perturbation_dists)
        print (f"New trajectory_{i_traj} created with {num_perturbed} atoms perturbed.")
        #copy_astr.to(filename='POSCAR_2', fmt='poscar')
        # Make supercell of the trajectory
        pnc_scell = np_box_obj.increase_pnc_size(copy_astr, size=scell_size,
                                                    alter_cells=True)
        traj_filename = candidate_dir + '/dump.{:09d}.txt'.format(i_traj * 1000)
        np_box_obj.write_trajectory(i_traj, pnc_scell, traj_filename)
        # To save memory
        del pnc_scell
        

    # Make one simulated SOQ with all the scells trajectory files
    soq_input_file = pnc_params['in_soq_path']
    with open(soq_input_file) as f:
        lines = f.readlines()
    lines[1] = 'endStep {}\n'.format(scells_for_one_soq*1000)
    # NOTE: leave the first 4 trajectories because they might be a bit too ordered
    lines[3] = 'initStep 10000\n' 
    to_file = candidate_dir + '/in.soq'
    with open(to_file, 'w') as f:
        f.writelines(lines)
    cmd = pnc_params['soq_exec_cmd']
    with open(to_file) as inp:
        sp.Popen(cmd.split(), stdout=sp.PIPE, stdin=inp).communicate()

    # Optimize the rescaling factor (rf)
    init_rf = [0.02]
    """
    opt_rf = minimize(get_chi_squared_stat, init_rf,
                                args=(candidate_dir, pnc_params),
                                method='L-BFGS-B',
                                bounds=[(0.011, 0.033)],
                                options={'maxiter': 10000})
    chi_squared_stat = get_chi_squared_stat(opt_rf.x, candidate_dir,
                                            pnc_params, plot=True)
    """
    opt_rf = minimize(soq_chi2_savgol, init_rf,
                                args=(candidate_dir, pnc_params),
                                method='L-BFGS-B',
                                bounds=[(0.011, 0.033)],
                                options={'maxiter': 10000})
    chi_squared_stat = soq_chi2_savgol(opt_rf.x, candidate_dir, pnc_params, plot=True)

    with open(candidate_dir+'/output.txt', 'w') as f:
        f.write('Optimized RF: {}\n'.format(opt_rf.x))
        f.write('Chi-squared stat: {}\n'.format(chi_squared_stat))

    declutter(candidate_dir)

    return chi_squared_stat

def declutter(candidate_dir):
    curr_dir = os.getcwd()
    
    os.chdir(candidate_dir)
    with open('params.txt') as f:
        param_lines = f.readlines()
    with open('output.txt') as f:
        out_lines = f.readlines()
    with open('POSCAR_NPbox2') as f:
        pos_lines = f.readlines()
    
    lines = param_lines
    lines.append('--------------------\n')
    lines += out_lines
    lines.append('--------------------\n')
    lines += pos_lines
    with open('candidate_info.txt', 'w') as f:
        f.writelines(lines)
    
    # Remove all dump files and declutter
    clutter = glob.glob("./dump*")
    clutter += ['params.txt', 'output.txt', 'POSCAR_NPbox2', 'in.soq']
    for f in clutter:
        os.remove(f)
    
    os.chdir(curr_dir)
    
######################################
######################################

import os, yaml
#from run_ops import *

# dask import
from dask_jobqueue import SLURMCluster, PBSCluster
from dask.distributed import Client, LocalCluster

# change worker unresponsive time to 3h (Assuming max elapsed time for one calc)
import dask
import dask.distributed
dask.config.set({'distributed.comm.timeouts.tcp': '5h'})

#from grid_trajectories import *

input_file = 'pnc_input_test_1.yaml'
main_path = os.getcwd()
if not os.path.exists(main_path + '/calcs'):
    os.mkdir(main_path + '/calcs')
else:
    os.rename(main_path + '/calcs', main_path + '/calcs_old')
    os.mkdir(main_path + '/calcs')
    print('calcs directory already exists. Renamed to calcs_old')

# Read inputs from the yaml file
with open(input_file) as f:
    i_dict = yaml.load(f, Loader=yaml.FullLoader)
    i_dict['main_path'] = main_path

data_file = main_path + '/data_file.txt'
with open(data_file, 'w') as f:
    f.write('Label, Chi_stat\n\n')

#### All inputs here ####
initial_population = 30
total_population = 1000
max_workers = 40
i_dict['initial_population'] = initial_population
i_dict['total_population'] = total_population

reg_id = register_id()

cluster_job = PBSCluster(cores=1,
                         memory="20gb",
                         account='cnm80881', ### Enter the project number
                         walltime='16:00:00',
                         interface='ib0',
                         job_extra_directives=['-l nodes=1:ppn=1:gen6'],
                         job_directives_skip=['-l select=1'])

cluster_job.scale(jobs=max_workers) # number of parallel jobs
client  = Client(cluster_job)

def full_eval(cand, pnc_params):
    # Evaluate the candidate -
    cand.chi_stat  = get_candidate_soq_residual_V2(cand, pnc_params)
    gc.collect()
    return cand

evald_futures = []
all_candidates = []
working_jobs = get_working_jobs(evald_futures)

pnc_params = get_random_param_set(i_dict)
pnc_params['main_path'] = main_path
cand = candidate(reg_id, pnc_params)
# Evaluate the candidate -
out = client.submit(full_eval, cand, pnc_params)
evald_futures.append(out)

while len(all_candidates) < initial_population:
    working_jobs = get_working_jobs(evald_futures)
    while working_jobs < max_workers:
        # Get random parameter set
        pnc_params = get_random_param_set(i_dict)
        if pnc_params is not None:
            pnc_params['main_path'] = main_path
            cand = candidate(reg_id, pnc_params)
            # Evaluate the candidate -
            out = client.submit(full_eval, cand, pnc_params)
            evald_futures.append(out)
        else:
            print ('Master is too fast! Waiting 5 seconds for workers to catch up..')
            time.sleep(5)
        evald_futures, all_candidates = update_pool(evald_futures,
                                                    all_candidates,
                                                    data_file)
        working_jobs = get_working_jobs(evald_futures)



update_selection_probs(all_candidates)
print ("Initial population completed with {} candidates".format(initial_population))

while len(all_candidates) < total_population-initial_population:
    working_jobs = get_working_jobs(evald_futures)
    while working_jobs < max_workers:

        # Get GA parameter set
        random_mode = np.random.choice(['avg', 'crossover', 'exploit'])
        pnc_params = get_child_param_set(all_candidates, i_dict, mode=random_mode)

        if pnc_params is not None:
            pnc_params['main_path'] = main_path
            cand = candidate(reg_id, pnc_params)
            print (f"Candidate label: {cand.label}")
            # Evaluate the candidate -
            out = client.submit(full_eval, cand, pnc_params)
            evald_futures.append(out)
            if len(evald_futures) + get_working_jobs(evald_futures) > 2*max_workers:
                time.sleep(5)
        else:
            print ('Master is too fast! Waiting 5 seconds for workers to catch up..')
            time.sleep(5)
        evald_futures, all_candidates = update_pool(evald_futures,
                                                    all_candidates,
                                                    data_file)
        update_selection_probs(all_candidates)
        working_jobs = get_working_jobs(evald_futures)


while len(evald_futures) > 0 or working_jobs > 0:
    evald_futures, all_candidates = update_pool(evald_futures,
                                                all_candidates,
                                                data_file)
    working_jobs = get_working_jobs(evald_futures)
print ('Done!!')

