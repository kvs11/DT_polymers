import os, yaml
from run_ops import *

# dask import
from dask_jobqueue import SLURMCluster, PBSCluster
from dask.distributed import Client, LocalCluster

# change worker unresponsive time to 3h (Assuming max elapsed time for one calc)
import dask
import dask.distributed
dask.config.set({'distributed.comm.timeouts.tcp': '3h'})


input_file = 'pnc_input_1.yaml'
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

initial_population = 20
total_population = 300
max_workers = 2
reg_id = register_id()

cluster_job = PBSCluster(cores=1,
                         memory="10GB",
                         project='cnm72851', ### Enter the project number
                         walltime='24:00:00',
                         interface='ib0',
                         job_extra=['-l nodes=1:ppn=1:gen6'],
                         header_skip=['-l select=1'])

cluster_job.scale(jobs=max_workers) # number of parallel jobs
client  = Client(cluster_job)

def full_eval(cand):
    # Evaluate the candidate -
    cand.chi_stat  = get_candidate_soq_residual(cand)
    return cand

evald_futures = []
all_candidates = []
working_jobs = get_working_jobs(evald_futures)

while len(all_candidates) < initial_population:
    working_jobs = get_working_jobs(evald_futures)
    while working_jobs < max_workers + 10:
        # Get random parameter set
        pnc_params = get_random_param_set(i_dict)
        pnc_params['main_path'] = main_path
        cand = candidate(reg_id, pnc_params)
        # Evaluate the candidate -
        out = client.submit(full_eval, cand)
        evald_futures.append(out)
        evald_futures, all_candidates = update_pool(evald_futures,
                                                    all_candidates, 
                                                    data_file)
        working_jobs = get_working_jobs(evald_futures)



update_selection_probs(all_candidates)
print ("Initial population completed with {} candidates".format(initial_population))

while len(all_candidates) < total_population-initial_population:
    working_jobs = get_working_jobs(evald_futures)
    while working_jobs < max_workers + 10:
        # Get GA parameter set
        pnc_params = get_param_set_GA(all_candidates, i_dict)
        pnc_params['main_path'] = main_path
        cand = candidate(reg_id, pnc_params)
        # Evaluate the candidate -
        out = client.submit(full_eval, cand)
        evald_futures.append(out)
        evald_futures, all_candidates = update_pool(evald_futures,
                                                    all_candidates, 
                                                    data_file)
        working_jobs = get_working_jobs(evald_futures)

while len(evald_futures) > 0:
    evald_futures, all_candidates = update_pool(evald_futures,
                                                all_candidates, 
                                                data_file)

print ('Done!!')