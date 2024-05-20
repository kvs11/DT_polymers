import numpy as np 
import random

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

    if 'num_clusters' not in fixed_params.keys() and \
                'num_NPs_per_cluster' not in fixed_params.keys():
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
    
    # Custom scaler to avoid outliers influence
    # Same as MinMax scaler but instead of Max we use 75th percentile
    chi2_min = chi_squared_values.min()
    chi2_max = np.percentile(chi_squared_values, 75)
    scaled_chi2_values = (chi_squared_values - chi2_min)/(chi2_max - chi2_min)
    selection_probs = 1 - scaled_chi2_values
    selection_probs[selection_probs < 0] = 0

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

def get_a_parent(all_candidates, initial_population):
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
            if len(all_candidates) < initial_population:
                print (f"Length of all_candidates: {len(all_candidates)} is low. "\
                        "Wait till new candidates are evaluated.")
                return None
            # remove parent from good_pool and continue
            print (f"Candidate {parent.label} reached maximum times chosen. Removing from all_candidates..")
            all_candidates.remove(parent)
            continue
        if parent.selection_prob:
            if random.random() < parent.selection_prob:
                return parent
    if num_attempts >= 100:
        print (f"get_a_parent could not get viable parent in 100 attempts.")
        return None

def get_two_parents(all_candidates, initial_population):
    cand_1 = get_a_parent(all_candidates, initial_population)
    # Setting a maximum number of attempts to create parents to avoid exhaustively 
    # creating new set of params while generated ones are being evaluated 
    num_attempts = 0
    while num_attempts < 10:
        num_attempts += 1
        cand_2 = get_a_parent(all_candidates, initial_population)
        if cand_1 is None or cand_2 is None:
            print ("get_a_parent returned None!!")
            return None
        if cand_1.label != cand_2.label:
            break
    if num_attempts >= 10:
        print ("Could not find second different parent in 10 attempts.")
        return None
    return cand_1, cand_2

def get_child_param_set(all_candidates, i_dict, mode='avg'):
    """
    """
    fixed_params = i_dict['pnc_fixed_params']
    variable_params = i_dict['pnc_variables']
    initial_population = i_dict['initial_population']

    cand_1, cand_2 = get_two_parents(all_candidates, initial_population)

    # get the new parameters
    new_params = {}
    if mode == 'avg': 
        for key in variable_params.keys():
            new_params[key] = (cand_1.params[key] + cand_2.params[key])/2
        cand_1.times_chosen_as_parent += 1
        cand_2.times_chosen_as_parent += 1
        print (f"New candidate is created by 'avg': {cand_1.label}, {cand_2.label}")

    elif mode == 'crossover':
        keys = list(variable_params.keys())
        np.random.shuffle(keys)
        for i, key in enumerate(keys):
            if i%2 == 0:
                new_params[key] = cand_1.params[key]
            else:
                new_params[key] = cand_2.params[key]
        cand_1.times_chosen_as_parent += 1
        cand_2.times_chosen_as_parent += 1
        print (f"New candidate is created by 'crossover': {cand_1.label}, {cand_2.label}")

    elif mode == 'exploit':
        for key in variable_params.keys():
            variable = cand_1.params[key]
            new_params[key] = np.random.normal(loc=variable, scale=variable*0.2)
        cand_1.times_chosen_as_parent += 1
        print (f"New candidate is created by 'exploit': {cand_1.label}")

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

    return {**fixed_params, **new_params}                         
