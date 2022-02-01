from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import \
    RotationTransformation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import os, random
import copy
import pandas as pd
import numpy as np
from math import pi
from numpy.random import uniform as unif
from concurrent.futures import ProcessPoolExecutor
# dask import
from dask_jobqueue import SLURMCluster, PBSCluster
from dask.distributed import Client
# change worker unresponsive time to 3h (Assuming max elapsed time for one calc)
import dask
import dask.distributed
dask.config.set({'distributed.comm.timeouts.tcp': '3h'})

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
        """
        found = False
        while not found:
            d = np.random.normal(loc=self.mean_dia_NP, scale=self.sigma_dia_NP)
            if self.min_dia_NP < d < self.max_dia_NP:
                found = True
        return d

    def get_box_latt(self):
        """
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
        while NPs_added < NPs_needed - 1 and new_loc_tries < NPs_needed+100:
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

    def write_trajectory(self, label, cart_coords, filename):
        """
        Get the structure in the form of LAMMPS trajectory,
        which is to be used as input for the soq/rdf code (by Jan Michael)
        """
        # TODO: using hard coded 10000 as timestep, change if needed
        time_step = label * 1000

        #if len(cart_coords) != n_NPs:
        #    print ('Warning: Required no. of NPs locs are not obtained')
        n_atoms = len(cart_coords)

        box_half_len = self.box_latt.a / 2
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

    def increase_pnc_size(self, init_astr, size=2):
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
        abc_sets = np.array([[1 ,0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
                             [0, 1, 1], [1, 0, 1], [1, 1, 1]])
        size_2_carts = list(init_carts)
        if size==2:
            for i in range(7):
                translated_astr = self.random_translation(init_astr)
                rotated_astr = self.random_rotation(translated_astr)
                new_carts = rotated_astr.cart_coords
                new_abc = abc_sets[i]
                new_min_xyz = init_min_xyz + abc * new_abc
                curr_min_xyz = new_carts.min(axis=0)
                translate_xyz = new_min_xyz - curr_min_xyz
                new_carts = new_carts + translate_xyz
                size_2_carts += list(new_carts)

        size_2_latt = init_astr.lattice.matrix * 2
        size_2_sps = ['Li' for i in range(len(size_2_carts))]
        size_2_astr = Structure(size_2_latt, size_2_sps, size_2_carts,
                                coords_are_cartesian=True)

        size_2_astr.sort()

        return size_2_astr

###########################################################################

np_box_params = {'mean_dia_NP' : 2.5,
                'sigma_dia_NP' : 0.1,
                'min_dia_NP' : 2.4,
                'max_dia_NP' : 2.6,
                'min_gap_NP' : 0.7,
                'vol_frac'   : 0.1,
                'box_len'    : 20,
                'num_clusters': 2,
                'num_NPs_per_cluster': 8,
                'cluster_shape': 'blob',
                'min_gap_NP_in_cluster': 0.2}
np_box_obj = nanoparticles_box(np_box_params)

label, rand_n = 11, 123
dia_NPs, pnc_astr = np_box_obj.get_rand_pnc_traj(label, rand_n,
                                                 write_trajectory=False,
                                                 return_structure=True)
size_2_pnc = np_box_obj.increase_pnc_size(pnc_astr, size=2)
size_4_pnc = np_box_obj.increase_pnc_size(size_2_pnc, size=2)
