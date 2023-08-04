

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
