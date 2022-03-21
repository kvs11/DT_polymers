###########################################################################
from pymatgen.core.structure import Structure

# define the function to be looped
def eval_one_soq(soq_label, pnc_params):
    """
    """
    rand_seed = np.random.randint(1000000)
    scell_size = pnc_params['scell_size']
    np_box_obj = nanoparticles_box(pnc_params)
    # Make one PNC unit cell
    dia_NPs, pnc_astr = np_box_obj.get_rand_pnc_traj(label, rand_n,
                                                     write_trajectory=False,
                                                     return_structure=True)

    # Make a directory for the candidate -
    # For one candidate SOQ with one label, need 25 PNCs
    candidate_dir = pnc_params['main_path'] + '/calcs/{}'.format(soq_label)
    os.mkdir(candidate_dir)
    os.chdir(candidate_dir)

    # Make "scells_for_one_soq" supercells with alterations
    scells_for_one_soq = pnc_params['scells_for_one_soq']
    for i_traj in range(1, scells_for_one_soq+1):
        pnc_scell = np_box_obj.increase_pnc_size(pnc_astr, size=scell_size,
                                                 alter_cells=True)
        traj_filename = cadidate_dir + '/dump.{:09d}.txt'.format(i_traj * 1000)
        np_box_obj.write_trajectory(i_traj, pnc_scell, traj_filename)

    # Make one simulated SOQ with all the scells trajectory files
    soq_input_file = pnc_params['in_soq_path']
    with open(soq_input_file) as f:
        lines = f.readlines()
    lines[1] = 'endStep {}\n'.format(scells_for_one_soq*1000)
    lines[3] = 'initStep 1000'
    to_file = cadidate_dir + '/in.soq'
    with open(to_file, 'w') as f:
        f.writelines(lines)
    cmd = pnc_params['soq_exec_cmd']
    with open(to_file) as inp:
        sp.Popen(cmd.split(), stdout=sp.PIPE, stdin=inp).communicate()

    sim_soq_path = candidate_dir + '/soq.txt'
    # Optimize the rescaling factor (rf)
    init_rf = [0.03]
    exp_soq_file = 'exp_soq.txt'
    fn_args = [candidate_dir, exp_soq_file]
    opt_rf = minimize(get_chi_squared_stat, init_rf, args=fn_args)

    chi_squared_stat = get_chi_squared_stat(opt_rf.x, candidate_dir,
                                            exp_soq_file)
