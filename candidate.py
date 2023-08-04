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
