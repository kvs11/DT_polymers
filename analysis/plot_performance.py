from cProfile import label
import os
import subprocess as sp
import scipy.optimize
from scipy.optimize import minimize
import numpy as np
import shutil

from scipy.interpolate import InterpolatedUnivariateSpline
from math import pi
import matplotlib.pyplot as plt

def get_xy(f):
    with open(f) as ff:
        lines = ff.readlines()
        x = [float(i.split()[0]) for i in lines]
        y = [float(i.split()[1]) for i in lines]
        return np.array([x[1:], y[1:]])

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

def get_params(params_file):
    """
    """
    with  open(params_file) as f:
        lines = f.readlines()

    pnc_params = {}
    for ll in lines:
        key = ll.split()[0]
        val = " ".join(ll.split()[1:])
        pnc_params[key] = val

    pnc_params['qrange'] = [float(i) for i in pnc_params['qrange'][1:-1].split(',')]
    pnc_params['weights'] = [float(i) for i in pnc_params['weights'][1:-1].split(',')]

    keys = ['sigma_dia_NP', 'min_dia_NP', 'max_dia_NP', 'box_len', 'scell_size',
            'scells_for_one_soq', 'q_tol', 'mean_dia_NP', 'min_gap_NP', 'vol_frac',
            'num_clusters', 'num_NPs_per_cluster', 'min_gap_NP_in_cluster']
    for k in keys:
        try:
            pnc_params[k] = int(pnc_params[k])
        except:
            pnc_params[k] = float(pnc_params[k])
    pnc_params['main_path'] = '/sandbox/vkolluru/DT_polymers/develop/Jan_2022/workflow_tests/grid_trajectories/test_1'
    return pnc_params
#########################

main_path = os.getcwd()
# 1. Get inds 
with open('data_file.txt') as f:
    lines = f.readlines()
lines = lines[2:]

all_inds, all_res = [], []
for l in lines:
    i, r = l.split()
    if float(r) == -1:
        continue
    all_inds.append(i)
    all_res.append(float(r))

sorted_inds = [i for _, i in sorted(zip(all_res, all_inds))]
sorted_res = all_res.copy()
sorted_res.sort()

intervals = [i*10 for i in range(3, 31)]
sel_inds, sel_res = [], []
for interval in intervals:
    for i, res in enumerate(sorted_res):
        if res > interval:
            sel_inds.append(sorted_inds[i])
            sel_res.append(res)
            break

new_sel_inds, new_sel_res = [], []
for i, ind in enumerate(sel_inds):
    if i > 0:
        if not sel_inds[i-1] == ind:
            new_sel_inds.append(ind)
            new_sel_res.append(sel_res[i])

# 2. Take soq.txt & get the positions of 1st peak, 1st trough and 2nd peak
calcs_dir = '/sandbox/vkolluru/DT_polymers/develop/Jan_2022/workflow_tests/grid_trajectories/test_1/calcs'
analysis_dir = '/sandbox/vkolluru/DT_polymers/develop/Jan_2022/workflow_tests/grid_trajectories/test_1/analysis'

# create empty data lists for keys in the following order:
# label, w_chi2, mean_dia_NP, min_gap_NP, vol_frac, num_clusters, num_NPs_per_cluster, min_gap_NP_in_cluster
data_lists = [[], [], [], [], [], [], [], []]
peak_dists = []
for d, chi2 in zip(sorted_inds, sorted_res): 
    cand_dir = calcs_dir + '/' + d
    os.chdir(cand_dir)
    sim_q, sim_soq = get_xy('soq.txt')

    # Fit a polynomial of degree 15 to the simualted soq data
    coeffs = np.polyfit(sim_q, sim_soq, 12)
    fitted_soq = np.polyval(coeffs, sim_q)

    # get q_extrema and soq_extrema
    q_extrema, soq_extrema = get_q_soq_extrema(sim_q[300:], fitted_soq[300:], skip=1)

    #plt.scatter(sim_q, sim_soq)
    #for i in range(3):
    #    plt.axvline(q_extrema[i])
    #plt.savefig('simd_soq.png')
    #plt.xlabel(r'q($\sigma^{-1}$)', fontsize=14)
    #plt.ylabel('S(q)', fontsize=14)
    #plt.close()

    # 3. Convert q to r (Ang.)
    q_to_dists = [2*pi/i for i in q_extrema]
    peak_dists.append(q_to_dists[:3])

    # 4. Take params.txt and get pnc params
    pnc_params = get_params('params.txt')
    data_lists[0].append(d)
    data_lists[1].append(chi2)
    data_lists[2].append(pnc_params['mean_dia_NP'])
    data_lists[3].append(pnc_params['min_gap_NP'])
    data_lists[4].append(pnc_params['vol_frac'])
    data_lists[5].append(pnc_params['num_clusters'])
    data_lists[6].append(pnc_params['num_NPs_per_cluster'])
    data_lists[7].append(pnc_params['min_gap_NP_in_cluster'])

    # 5. Copy the soq_plot.png and simd_soq.png to analysis_dir
    to1 = analysis_dir + '/soq_plot_{}.png'.format(d)
    to2 = analysis_dir + '/simd_soq_{}.png'.format(d)
    #shutil.copyfile('soq_plot.png', to1)
    #shutil.copyfile('simd_soq.png', to2)
# Decide what plots to make
os.chdir(main_path)

#### Plot 1 #####
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
chi2_vals = data_lists[1]
axes[0, 0].scatter(chi2_vals, data_lists[2], marker='o', s=50, alpha=0.25)
#axes[0, 0].plot(chi2_vals, data_lists[2], label='mean_dia_NP')
axes[0, 0].set_title('mean_dia_NP')
axes[0, 0].set_ylim(1.9, 2.55)
axes[0, 0].set_xlim(0, 400)

axes[0, 1].scatter(chi2_vals, data_lists[3], marker='o', s=50, alpha=0.25)
#axes[0, 1].plot(chi2_vals, data_lists[3], label='min_gap_NP')
axes[0, 1].set_title('min_gap_NP')
axes[0, 1].set_ylim(0.29, 0.51)
axes[0, 1].set_xlim(0, 400)

axes[0, 2].scatter(chi2_vals, data_lists[4], marker='o', s=50, alpha=0.25)
#axes[0, 2].plot(chi2_vals, data_lists[4], label='vol_frac')
axes[0, 2].set_title('vol_frac')
axes[0, 2].set_ylim(0.05, 0.27)
axes[0, 2].set_xlim(0, 400)

axes[1, 0].scatter(chi2_vals, data_lists[5], marker='o', s=50, alpha=0.25)
#axes[1, 0].plot(chi2_vals, data_lists[5], label='num_clusters')
axes[1, 0].set_title('num_clusters')
axes[1, 0].set_ylim(1.9, 4.1)
axes[1, 0].set_xlabel('Weighted chi2', fontsize=14)
axes[1, 0].set_xlim(0, 400)

axes[1, 1].scatter(chi2_vals, data_lists[6], marker='o', s=50, alpha=0.25)
#axes[1, 1].plot(chi2_vals, data_lists[6], label='num_atoms_per_cluster')
axes[1, 1].set_title('num_atoms_per_cluster')
axes[1, 1].set_ylim(7.8, 15.2)
axes[1, 1].set_xlabel('Weighted chi2', fontsize=14)
axes[1, 1].set_xlim(0, 400)

axes[1, 2].scatter(chi2_vals, data_lists[7], marker='o', s=50, alpha=0.25)
#axes[1, 2].plot(chi2_vals, data_lists[7], label='min_gap_NPs_in_cluster')
axes[1, 2].set_xlabel('Weighted chi2', fontsize=14)
axes[1, 2].set_title('min_gap_NPs_in_cluster')
axes[1, 2].set_ylim(0.1, 0.25)
axes[1, 2].set_xlim(0, 400)

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig('all_data_lists_test4.png')
plt.close()

#### Plot 2 #####
peak_dists = np.array(peak_dists).T
plt.scatter(chi2_vals, peak_dists[0], c='orange', s=50, marker='o', alpha=0.25)
plt.scatter(chi2_vals, peak_dists[1], c='blue', s=50, marker='o', alpha=0.25)
plt.scatter(chi2_vals, peak_dists[2], c='green', s=50, marker='o', alpha=0.25)
plt.xlabel('W_chi2', fontsize=14)
plt.ylabel(r'Distance ($\AA$)', fontsize=14)
plt.legend(['1st peak', '1st trough', '2nd peak'])
plt.savefig('peak_dists.png')
plt.close()



# Fit a polynomial of degree 5 to the simualted soq data
sim_q, sim_soq = get_xy('soq.txt')
coeffs = np.polyfit(sim_q, sim_soq, 15)
fitted_soq = np.polyval(coeffs, sim_q)
plt.scatter(sim_q, sim_soq, c='blue')
plt.plot(sim_q, fitted_soq, c='red')
plt.savefig('ttt.png')
plt.close()



plt.scatter(exp_new_x, exp_new_y, c='blue')
plt.scatter(exp_new_x, fitted_exp_new_y, c='red')
plt.savefig('qqq.png')
plt.close()

##################################################
chi2_vals = data_lists[1]
# combine mean_dia_NP and vol_frac 
plt.figure(figsize=(8, 6))
plt.scatter(chi2_vals, data_lists[4], c=data_lists[2], cmap='hot',
            marker='o', s=50) #, alpha=0.25
plt.colorbar(label='mean_dia_NP')
plt.xlim(0, 1000)
plt.xlabel('W_chi2', fontsize=14)
plt.ylabel('vol_frac_', fontsize=14)
plt.savefig('vol_frac_mean_dia_NP.png')
plt.close()


# min_gap_NP and vol_frac 
##################################################
chi2_vals = data_lists[1]
# combine mean_dia_NP and vol_frac 
plt.figure(figsize=(8, 6))
plt.scatter(chi2_vals, data_lists[4], c=data_lists[3], cmap='hot',
            marker='o', s=50) #, alpha=0.25
plt.colorbar(label='min_gap_NP')
plt.xlim(0, 1000)
plt.xlabel('W_chi2', fontsize=14)
plt.ylabel('vol_frac', fontsize=14)
plt.savefig('vol_frac_min_gap_NP.png')
plt.close()

# mean_dia_NP and min_gap_NP
chi2_vals = data_lists[1]
# combine mean_dia_NP and vol_frac 
plt.figure(figsize=(8, 6))
plt.scatter(chi2_vals, data_lists[3], c=data_lists[2], cmap='hot',
            marker='o', s=50) #, alpha=0.25
plt.colorbar(label='min_gap_NP')
plt.xlim(0, 1000)
plt.xlabel('W_chi2', fontsize=14)
plt.ylabel('mean_dia_NP', fontsize=14)
plt.savefig('min_gap_mean_dia_NP.png')
plt.close()