import os
import time
#from shutil import copy as shutilcopy
import subprocess as sp
import numpy as np
import random
from math import pi
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize
from scipy.optimize import minimize
from NPs_box import nanoparticles_box

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
    lines[3] = 'initStep 1000'
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

    # Convert q(sigma-1) --> q(Å-1)
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
        if i_traj < 4 and time.time() - start_time > 100:
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
    chi_squared_values = chi_squared_values.reshape(-1, 1)
    # normalize using MinMaxScaler
    scaler = MinMaxScaler()
    selection_probs = scaler.fit_transform(
                        1-scaler.fit_transform(chi_squared_values))
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

def get_a_parent(all_candidates):
    """
    Returns exactly one parent
    """
    done = False
    while not done:
        # randomly choose a parent
        parent = random.choice(all_candidates)
        if parent.times_chosen_as_parent > parent.max_times_as_parent:
            # remove parent from good_pool and continue
            all_candidates.remove(parent)
            continue
        if parent.selection_prob:
            if random.random() < parent.selection_prob:
                return parent

def get_param_set_GA(all_candidates, i_dict):
    """
    """
    fixed_params = i_dict['pnc_fixed_params']
    variable_params = i_dict['pnc_variables']

    cand_1 = get_a_parent(all_candidates)
    while True:
        cand_2 = get_a_parent(all_candidates)
        if cand_1.label != cand_2.label:
            break
    # get the new parameters
    new_params = {}
    for key in variable_params.keys():
        new_params[key] = (cand_1.params[key] + cand_2.params[key])/2
    for key in ['num_clusters', 'num_NPs_per_cluster']:
        if not isinstance(new_params[key], int):
            try:
                new_params[key] = int(new_params[key])
            except:
                int_param = int(new_params[key])
                new_params[key] = np.random.choice([int_param, int_param+1])
    while new_params['num_clusters'] * new_params['num_NPs_per_cluster'] > 60:
        new_params['num_NPs_per_cluster'] += -1

    cand_1.times_chosen_as_parent += 1
    cand_2.times_chosen_as_parent += 1
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
    