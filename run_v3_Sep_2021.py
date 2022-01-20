import os
import numpy as np


"""
read input Parameters:
Constant parameters required "for a whole run":
        > min dia NP ; max dia NP
        > min vol frac ; max vol frac
        > min box size ; max box size
        > min n_NPs : max n_NPs

Optimize following variable parameters:
        > mean dia NP, sigma dia NP
        > min gap NP
        > volume fraction

To get "One model":
    Make 100 NPs_boxes
    Get one sim_soq
    Compare the sim_soq with target_soq

Goal: Make several models (say 1000) and get optimum solution for variable
      parameters

Question: Which optimization to use? Simple multi-variate optimization? (with
          or without constraints)?

-----------------------------------------------------------------------
To use simple multivariate optmization -

>>>>>>> Get all parameters
1. Function to get all required params, constant and variable.

>>>>>>> Cost Function to optmize
2. Function which returns a scalar value
    > make 100 PNCs with the above params
    > get sim_soq
    > get RMSE for the fit with discrete values
        ** Write function to get RMSE

>>>>>>> Do scipy.optimize.minimize with above function

"""
# general functions
def sphere_vol(radius):
    return 4/3 * pi * radius**3

# one loop

# get all fixed params
n_NPs = 1000 # fix n_NPs

min_dia_NP = 2.0
max_dia_NP = 2.5

min_min_gap_NP = 0.1
max_min_gap_NP = 0.7

min_vol_frac = 0.05
max_vol_frac = 0.35

# get init values for all variable params
mean_dia_NP = 2.25
sigma_dia_NP = 0.3
min_gap_NP = 0.3
vol_frac = 0.2
# get box_len for current vol_frac and n_NPs
box_len = (vol_NPs / vol_frac)**(1/3)

# get min box length and max box length (just for information)
vol_NP = sphere_vol(mean_dia_NP/2)
vol_NPs = n_NPs * vol_NP
max_box_len = (vol_NPs / min_vol_frac)**(1/3)
min_box_len = (vol_NPs / max_vol_frac)**(1/3)

print ('For the given range of volume fractions & {0} NPs, box length varies '
    'between {1:.2f} and {2:.2f} units'.format(n_NPs, min_box_len, max_box_len))

# make 50 PNCs
np_box_params = {'mean_dia_NP' : mean_dia_NP,
                 'sigma_dia_NP': sigma_dia_NP,
                 'min_dia_NP'  : min_dia_NP,
                 'max_dia_NP'  : max_dia_NP,
                 'min_gap_NP'  : min_gap_NP,
                 'vol_frac'    : vol_frac,
                 'box_len'     : box_len}
np_box_obj = nanoparticles_box(np_box_params)
total = 48 # chosen 48 instead of 50 because divisible by total workers 16
seed_rands = np.random.randint(100, 1000000, size=total)
executor = ProcessPoolExecutor(max_workers=16)

futures = []
for label, rand_n in zip(range(total), seed_rands):
    out = executor.submit(np_box_obj.get_rand_pnc_traj, label, rand_n)
    futures.append(out)

labels, diaNPs_trajs = [], []
for f in futures:
    diaNPs_trajs.append(f.result())

dia_arr = np.array(diaNPs_trajs)
df = pd.DataFrame(data=dia_arr, columns=['NP'+str(i+1) for i in \
                                        range(len(diaNPs_trajs[0]))])
df.to_csv('data_dia_{}vf.csv'.format(vol_frac))

# get simulated soq from the 50 PNCs
def get_sim_soq(cmd, in_soq_path):
    """
    """
    with open(in_soq_path) as inp:
        ff = open('full_log.txt', 'ab')
        soq_out = sp.Popen(cmd.split(), stderr=sp.STDOUT,
                                  stdout=sp.PIPE, stdin=inp)
        for line in soq_out.stdout:
            ff.write(line)
        soq_out.wait()
        ff.close()


# scalar residual function
def scalar_residual_funciton():
    """
    """

    pass

# scipy.minimize.optimize(scalar_residual_funciton, X)



















##
