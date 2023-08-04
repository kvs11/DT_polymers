import numpy as np
import os, random
from math import pi

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
