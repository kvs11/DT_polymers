from pymatgen.core.lattice import Lattice

import os, random
import pandas as pd
import numpy as np
from math import pi
from numpy.random import uniform as unif
from concurrent.futures import ProcessPoolExecutor

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

    def get_rand_frac_coords(self):
        """
        """
        random_coords = []
        dia_NPs = []
        new_loc_tries = 0
        NPs_added = 0
        while NPs_added < self.n_NPs - 1 and new_loc_tries < self.n_NPs+100:
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
    def get_rand_pnc_traj(self, label, rand_seed):
        """
        """
        np.random.seed(rand_seed)
        rand_fracs, dia_NPs = self.get_rand_frac_coords()
        cart_coords = self.box_latt.get_cartesian_coords(rand_fracs)
        traj_filename = 'dump.{:09d}.txt'.format(label * 1000)
        self.write_trajectory(label, cart_coords, traj_filename)
        #"Done {}".format(label)
        return dia_NPs

    def get_NP_coords_cluster(self, num_atoms, cluster_dia, shuffle_center=False):
        """
        Given maximum allowed diamter of a cluster, this function adds random
        coordinates in a chain like fashion connected to the previous added
        atom which satisfies distance constraints with other atoms present.

        Returns a list of cartesian coordinates

        Args:

        num_atoms (int) - number of atoms needed in the structure

        cluster_dia (float) - maximum diameter of the cluster
        """
        # start from origin
        old_NP_point = np.array([0, 0, 0])
        old_NP_dia = self.get_dia_NP()

        dia_NPs = []
        dia_NPs.append(old_NP_dia)


        rand_cart_coords = []
        coords_added = 0
        new_NP_point_attempt = 0
        while coords_added < num_atoms:
            new_NP_point = self.get_point_on_sphere(dia_NP/2)
            new_NP_dia = self.get_dia_NP()
            dist_NP = old_NP_dia/2 + new_NP_dia/2 + self.min_gap_cluster

            # returns None if the algo cannot add a new point in 500 attempts
            # if the cluster_diameter is too small, this algo hangs trying to
            # add new point
            new_NP_point_attempt += 1
            if new_NP_point_attempt > 1000:
                return None

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
            dist_check = True
            while dist_check is True:
                for each_NP in coords_in_new_sphere:
                    dist = each_NP[1]
                    dia_NP2 = dia_NPs[each_NP[2]]
                    if not dist > new_NP_dia/2 + dia_NP2/2 + self.min_gap_NP:
                        dist_check = False
            if not dist_check:
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

        # coords are cartesian
        return rand_cart_coords

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

###########################################################################



vfs = [0.16, 0.2, 0.25, 0.3, 0.4, 0.5]

for vol_frac in vfs:
    dirname = 'vf_{}'.format(vol_frac)
    os.mkdir(dirname)
    os.chdir(dirname)
    np_box_params = {'mean_dia_NP' : 2.5,
                'sigma_dia_NP' : 1,
                'min_dia_NP' : 2.1,
                'max_dia_NP' : 3,
                'min_gap_NP' : 0.2,
                'vol_frac'   : vol_frac,
                'box_len'    : 50}
    np_box_obj = nanoparticles_box(np_box_params)

    total = 100
    seed_rands = np.random.randint(100, 1000000, size=total)
    executor = ProcessPoolExecutor(max_workers=16)

    futures = []
    for label, rand_n in zip(range(total), seed_rands):
        out = executor.submit(np_box_obj.get_rand_pnc_traj, label, rand_n)
        futures.append(out)

    labels, diaNPs_100 = [], []
    for f in futures:
        diaNPs_100.append(f.result())

    dia_arr = np.array(diaNPs_100)
    df = pd.DataFrame(data=dia_arr, columns=['NP'+str(i+1) for i in \
                                            range(len(diaNPs_100[0]))])
    df.to_csv('data_dia_{}vf.csv'.format(vol_frac))
    os.chdir('../')
