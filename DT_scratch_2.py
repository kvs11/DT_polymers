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

###########################################################################

np_box_params = {'mean_dia_NP' : 2.5,
                'sigma_dia_NP' : 0.1,
                'min_dia_NP' : 2.4,
                'max_dia_NP' : 2.6,
                'min_gap_NP' : 0.5,
                'vol_frac'   : 0.1,
                'box_len'    : 20,
                'num_clusters': 3,
                'num_NPs_per_cluster': 8,
                'cluster_shape': 'combination',
                'min_gap_NP_in_cluster': 0.2}
np_box_obj = nanoparticles_box(np_box_params)

label = 11
rand_n = np.random.randint(1000000)
import time
t1 = time.time()
dia_NPs, pnc_astr = np_box_obj.get_rand_pnc_traj(label, rand_n,
                                                 write_trajectory=False,
                                                 return_structure=True)

scell_sizes = [2, 3, 4, 5]

curr_dir = os.getcwd()
for size in scell_sizes:
    os.chdir(curr_dir)
    dirname = curr_dir + '/size_{}_pncs'.format(size)
    os.mkdir(dirname)
    os.chdir(dirname)
    # Make one supercell without alterations
    pnc_scell = np_box_obj.increase_pnc_size(pnc_astr, size=size,
                                             alter_cells=False)
    label = 0
    traj_filename = dirname + '/dump.{:09d}.txt'.format(label * 1000)
    np_box_obj.write_trajectory(label, pnc_scell, traj_filename)
    # Make 10 supercells with alterations
    for label in range(1, 11):
        pnc_scell = np_box_obj.increase_pnc_size(pnc_astr, size=size,
                                                 alter_cells=True)
        traj_filename = dirname + '/dump.{:09d}.txt'.format(label * 1000)
        np_box_obj.write_trajectory(label, pnc_scell, traj_filename)
