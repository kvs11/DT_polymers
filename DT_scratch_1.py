def get_NP_coords_cluster(np_box_obj, num_NPs, cluster_dia=None, shuffle_center=False):
    """
    Given maximum allowed diamter of a cluster, this function adds random
    coordinates in a chain like fashion connected to the previous added
    atom which satisfies distance constraints with other atoms present.

    Returns a list of cartesian coordinates

    Args:

    num_NPs (int) - number of atoms needed in the structure

    cluster_dia (float) - maximum diameter of the cluster
    """
    # start from origin
    old_NP_point = np.array([0, 0, 0])
    # Add first point
    rand_cart_coords = []
    rand_cart_coords.append(old_NP_point)
    coords_added = 1
    old_NP_dia = np_box_obj.get_dia_NP()
    dia_NPs = []
    dia_NPs.append(old_NP_dia)


    new_NP_point_attempt = 0
    while coords_added < num_NPs:
        new_NP_dia = np_box_obj.get_dia_NP()
        dist_NP = old_NP_dia/2 + new_NP_dia/2 + \
                            np_box_obj.min_gap_NP_in_cluster
        new_NP_point = np_box_obj.get_point_on_sphere(dist_NP)

        # returns None if the algo cannot add a new point in 500 attempts
        # if the cluster_diameter is too small, this algo hangs trying to
        # add new point
        new_NP_point_attempt += 1
        if new_NP_point_attempt > 1000:
            print (new_NP_point_attempt)
            break #return rand_cart_coords

        # translate the point near the old_NP_point
        new_NP_point = new_NP_point + old_NP_point

        # check if the translated point is within cluster diamter box
        if cluster_dia is not None:
            if not np.linalg.norm(new_NP_point) < cluster_dia/2:
                print ('magnitude of new point: ', np.linalg.norm(new_NP_point))
                continue

        # Get points within a sphere of max dist for second point
        max_dist = new_NP_dia/2 + np_box_obj.max_dia_NP/2 + np_box_obj.min_gap_NP
        rand_frac_coords = np_box_obj.box_latt.get_fractional_coords(
                                                    rand_cart_coords)
        coords_in_new_sphere = np_box_obj.box_latt.get_points_in_sphere_py(
                                rand_frac_coords, new_NP_point, max_dist)

        # check dist individually with each NP within the max_dist
        dist_check = 0
        for each_NP in coords_in_new_sphere:
            dist = each_NP[1]
            print (dist, dist_NP)
            dia_NP2 = dia_NPs[each_NP[2]]
            if dist < new_NP_dia/2 + dia_NP2/2 + \
                                np_box_obj.min_gap_NP_in_cluster - 0.1:
                dist_check += 1
                print (new_NP_point, rand_cart_coords)
                break
        if dist_check != 0:
            print ('Attempt {} failed'.format(new_NP_point_attempt))
            continue

        # add the new_NP_point and reset the no. of attempts
        dia_NPs.append(new_NP_dia)
        rand_cart_coords.append(new_NP_point)
        new_NP_point_attempt = 0
        old_NP_point = new_NP_point
        if shuffle_center is True:
            old_NP_point = rand_cart_coords[np.random.randint(
                                                len(rand_cart_coords))]
        coords_added += 1

    a = np_box_obj.box_latt.a
    rand_cart_coords = np.array(rand_cart_coords) + np.array([a/2, a/2, a/2])

    if len(rand_cart_coords) < num_NPs:
        return None
    # coords are cartesian
    return cc

###########################################################################
from pymatgen.core.structure import Structure
while True:
    for i in range(10):
        cc = get_NP_coords_cluster(np_box_obj, 10, cluster_dia=None, shuffle_center=True)
        s = Structure(np_box_obj.box_latt, ['Li' for i in range(len(cc))], cc, coords_are_cartesian=True)
        s.to(filename='POSCAR_sT{}'.format(i))

    for i in range(10):
        cc = get_NP_coords_cluster(np_box_obj, 10, cluster_dia=None, shuffle_center=False)
        s = Structure(np_box_obj.box_latt, ['Li' for i in range(len(cc))], cc, coords_are_cartesian=True)
        s.to(filename='POSCAR_sF{}'.format(i))

    for i in range(10):
        cc = get_NP_coords_cluster(np_box_obj, 10, cluster_dia=10, shuffle_center=False)
        s = Structure(np_box_obj.box_latt, ['Li' for i in range(len(cc))], cc, coords_are_cartesian=True)
        s.to(filename='POSCAR_cd10_sF{}'.format(i))

    for i in range(10):
        cc = get_NP_coords_cluster(np_box_obj, 10, cluster_dia=10, shuffle_center=False)
        s = Structure(np_box_obj.box_latt, ['Li' for i in range(len(cc))], cc, coords_are_cartesian=True)
        s.to(filename='POSCAR_cd10_sT{}'.format(i))
    break
###########################################################################

def put_n_clusters(np_box_obj, num_clusters, num_NPs_per_cluster, cluster_dia=None, shuffle_center=False):
    """
    """
    if num_clusters > 5:
        # TODO
        print ('num_clusters should be 5 or less.')

    all_clusters = []
    while len(all_clusters) < num_clusters:
        clus_carts = np_box_obj.get_NP_coords_cluster(
                            num_NPs_per_cluster, cluster_dia=cluster_dia, shuffle_center=shuffle_center)
        if clus_carts is not None:
            all_clusters.append(clus_carts)

    #### TEMP
    a = np_box_obj.box_latt.a
    if num_clusters == 1:
        clus_carts = all_clusters[0] + np.array([a/2, a/2, a/2])
        all_clus_carts = clus_carts
    if num_clusters == 2:
        clus_carts = all_clusters[0] + np.array([a/4, a/4, a/4])
        all_clus_carts = clus_carts
        clus_carts = all_clusters[1] + np.array([3*a/4, 3*a/4, 3*a/4])
        all_clus_carts = np.concatenate((all_clus_carts, clus_carts))
    if num_clusters >= 3:
        clus_carts = all_clusters[0] + np.array([a/2, a/2, 0])
        all_clus_carts = clus_carts
        clus_carts = all_clusters[1] + np.array([0, a/2, a/2])
        all_clus_carts = np.concatenate((all_clus_carts, clus_carts))
        clus_carts = all_clusters[2] + np.array([a/2, 0, a/2])
        all_clus_carts = np.concatenate((all_clus_carts, clus_carts))
    if num_clusters >= 4:
        clus_carts = all_clusters[3] + np.array([a/2, a/2, a/2])
        all_clus_carts = np.concatenate((all_clus_carts, clus_carts))
    if num_clusters == 5:
        clus_carts = all_clusters[4]
        all_clus_carts = np.concatenate((all_clus_carts, clus_carts))

    return all_clus_carts

while True:
    for i in range(10):
        cc = put_n_clusters(np_box_obj, 4, 10, cluster_dia=None, shuffle_center=True)
        s = Structure(np_box_obj.box_latt, ['Li' for i in range(len(cc))], cc, coords_are_cartesian=True)
        s.to(filename='POSCAR_sT{}'.format(i))

    for i in range(10):
        cc = put_n_clusters(np_box_obj, 4, 10, cluster_dia=None, shuffle_center=False)
        s = Structure(np_box_obj.box_latt, ['Li' for i in range(len(cc))], cc, coords_are_cartesian=True)
        s.to(filename='POSCAR_sF{}'.format(i))

    for i in range(10):
        cc = put_n_clusters(np_box_obj, 4, 10, cluster_dia=10, shuffle_center=False)
        s = Structure(np_box_obj.box_latt, ['Li' for i in range(len(cc))], cc, coords_are_cartesian=True)
        s.to(filename='POSCAR_cd10_sF{}'.format(i))

    for i in range(10):
        cc = put_n_clusters(np_box_obj, 4, 10, cluster_dia=10, shuffle_center=False)
        s = Structure(np_box_obj.box_latt, ['Li' for i in range(len(cc))], cc, coords_are_cartesian=True)
        s.to(filename='POSCAR_cd10_sT{}'.format(i))
    break
###########################################################################

def get_rand_frac_coords(np_box_obj, curr_NP_coords=None, coords_are_cartesian=True, curr_dia_NPs=None):
    """
    """
    random_coords = []
    dia_NPs = []
    NPs_needed = np_box_obj.n_NPs
    NPs_added = 0
    if curr_NP_coords is not None:
        random_coords = list(curr_NP_coords)
        if coords_are_cartesian:
            random_coords = list(np_box_obj.box_latt.get_fractional_coords(
                                                        curr_NP_coords))
        if curr_dia_NPs is None:
            dia_NPs = [np_box_obj.mean_dia_NP for i in range(len(random_coords))]
        else:
            dia_NPs = list(curr_dia_NPs)
        NPs_added = len(random_coords)
        NPs_needed = np_box_obj.n_NPs - NPs_added

    new_loc_tries = 0
    while NPs_added < NPs_needed - 1 and new_loc_tries < NPs_needed+100:
        new_loc_tries += 1
        if len(random_coords) == 0:
            new_fracs = [unif(0, 1), unif(0, 1), unif(0, 1)]
            dia_NP = np_box_obj.get_dia_NP()
            dia_NPs.append(dia_NP)
            random_coords.append(new_fracs) # frac coords added
            continue
        # Starting from second random coords, check min_dist_NP constraint
        # considering the periodic boundary condition (pbc)
        added_new_fracs = False
        num_tries = 0
        while not added_new_fracs and num_tries < 500:
            num_tries += 1
            # Get the next new_fracs
            new_fracs = [unif(0, 1), unif(0, 1), unif(0, 1)]
            new_carts = np_box_obj.box_latt.get_cartesian_coords(new_fracs)
            dia_NP = np_box_obj.get_dia_NP()
            # Get points within a sphere of max dist for second point
            max_dist = dia_NP/2 + np_box_obj.max_dia_NP/2 + np_box_obj.min_gap_NP
            coords_in_new_sphere = np_box_obj.box_latt.get_points_in_sphere_py(
                                    random_coords, new_carts, max_dist)

            # check dist individually with each NP within the max_dist
            dist_check = 0
            for each_NP in coords_in_new_sphere:
                dist = each_NP[1]
                dia_NP2 = dia_NPs[each_NP[2]]
                if not dist > dia_NP/2 + dia_NP2/2 + np_box_obj.min_gap_NP:
                    dist_check += 1
                    break
            if dist_check == 0:
                dia_NPs.append(dia_NP)
                random_coords.append(new_fracs)
                added_new_fracs = True
                NPs_added += 1

    return random_coords, dia_NPs
