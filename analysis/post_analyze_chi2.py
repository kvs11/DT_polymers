import os, yaml
import numpy as np 

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import minimize

import matplotlib.pyplot as plt


def get_xy(f):
    with open(f) as ff:
        lines = ff.readlines()
        x = [float(i.split()[0]) for i in lines]
        y = [float(i.split()[1]) for i in lines]
        return np.array([x[1:], y[1:]])

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

def get_q_soq_extrema(exp_q, exp_soq, skip=2):
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

def soq_chi2_savgol(rescaling_factor, pnc_params, plot=False):
    
    # Read sim_soq.txt
    sim_soq_file = 'soq.txt'
    sim_x, sim_y = get_xy(sim_soq_file)
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
        plt.savefig('soq_plot_3.png')
        plt.close()

    return weighted_chisquared
    
############################ Inputs ############################
pnc_params = {}
pnc_params["weights"] = [100, 100, 100]
pnc_params["q_tol"] = 0.005
exp_soq_file = '/sandbox/vkolluru/DT_polymers/develop/Jan_2022/workflow_tests/grid_trajectories/soq_Koga_27vf.txt'
calcs_path = '/sandbox/vkolluru/DT_polymers/develop/Jan_2022/workflow_tests/grid_trajectories/test_1/calcs/'
qrange = [0.015, 0.15]
rescaling_factor = 0.02
################################################################

# Read the experimental SOQ 
exp_q, exp_soq = get_xy(exp_soq_file)
# & Denoise it with Savitzky-Golay filter
denoised_exp_soq = savgol_filter(exp_soq, 51, 2) # window size 50 is good for exp soq

# Go to calcs/ directory
os.chdir(calcs_path)
# Go to model directory
all_models = os.listdir()

for candidate_dir in all_models:
    os.chdir(candidate_dir)

    # Optimize the rescaling factor (rf)
    init_rf = [0.02]
    opt_rf = minimize(soq_chi2_savgol, init_rf,
                                args=(pnc_params),
                                method='L-BFGS-B',
                                bounds=[(0.011, 0.033)],
                                options={'maxiter': 10000})
    chi_squared_stat = soq_chi2_savgol(opt_rf.x, pnc_params, plot=True)
    with open('output_2.txt', 'w') as f:
        f.write('\nOptimized RF: {}\n'.format(opt_rf.x))
        f.write('Chi-squared stat: {}\n'.format(chi_squared_stat))
    os.chdir(calcs_path)