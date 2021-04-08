from importlib import reload
import structure

structure = reload(structure)

polymer = structure.polymer

# Input parameters
n_NP = 5
dia_NP = 20
min_dist_NP = 23
chain_length = 25

# generate label
reg_id = structure.register_id()

polymer_parameters = {}
polymer_parameters['data_file'] = 'compressed_555.dat'
polymer_parameters['n_NP'] = n_NP
polymer_parameters['dia_NP'] = dia_NP
polymer_parameters['min_dist_NP'] = min_dist_NP
polymer_parameters['chain_length'] = chain_length
polymer_parameters['label'] = reg_id.create_id()

# Create polymer object
poly = polymer(polymer_parameters)
# get random coords for the NPs in the polymer matrix
random_coords = poly.get_NP_locs()

# get the mol_ids_to_remove to make cavities for the NPs
mol_ids_to_remove = poly.get_mol_ids_to_remove(random_coords)

# make data file and return the new_lmp_data of it
new_lmp_data = poly.make_data_file(mol_ids_to_remove, 
                                   random_coords)

# create a trajectory file for the new data file using new_lmp_data
path_to_dump = './'
poly.write_trajectory(new_lmp_data, path_to_dump)
new_lmp_data.structure.to(filename='POSCAR_test_3.vasp')

# test if this is successful
ss = new_lmp_data.structure
nps = ss.cart_coords[-5:]
for carts in nps:
    xx = ss.lattice.get_points_in_sphere(ss.frac_coords, carts, 
                                         poly.dia_NP/2)
    print (len(xx))
    
    
    
    
    
    
    
    #