def get_param_set_GA(all_candidates, i_dict):
    """
    """
    fixed_params = i_dict['pnc_fixed_params']
    variable_params = i_dict['pnc_variables']

    cand_1 = get_a_parent(all_candidates)
    # Setting a maximum number of attempts to create parents to avoid exhaustively 
    # creating new set of params while generated ones are being evaluated 
    num_attempts = 0
    while num_attempts < 10:
        num_attempts += 1
        cand_2 = get_a_parent(all_candidates)
        if cand_1 is None or cand_2 is None:
            return None
        if cand_1.label != cand_2.label:
            break
    if num_attempts >= 10:
        return None
    
    # get the new parameters
    new_params = {}
    for key in variable_params.keys():
        new_params[key] = (cand_1.params[key] + cand_2.params[key])/2

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

    cand_1.times_chosen_as_parent += 1
    cand_2.times_chosen_as_parent += 1
    print (f"New candidate is created from labels: {cand_1.label}, {cand_2.label}")
    return {**fixed_params, **new_params}

def get_a_parent(all_candidates):
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
            # remove parent from good_pool and continue
            all_candidates.remove(parent)
            continue
        if parent.selection_prob:
            if random.random() < parent.selection_prob:
                return parent
    if num_attempts >= 100:
        return None