import numpy as np
import xarray as xr
import glob

from utils import read_yaml_config
from makeRetCore import MakeRetrieval

# import general parameters and retrieval configuration
# read params
ret_type = 'tpt'

if ret_type == 'tpb':
    config = read_yaml_config("configs/config_" + ret_type + ".yaml")
    params = config['params']
    # read ret_specs
    ret_specs = read_yaml_config("ret_specs.yaml")
else:
    config = read_yaml_config("configs/config_" + ret_type + ".yaml")
    params = config['params']

# print(params)

# list files containing the radiative transfer output
rt_file_list1 = []
for i_list in np.arange(len(params['rt_paths'])):
    for j_list in np.arange(len(params['rt_patterns'])):
        rt_file_list0 = glob.glob(params['rt_paths'][i_list] + params['rt_patterns'][j_list])
        rt_file_list0.sort()
        rt_file_list1.append(rt_file_list0)

rt_file_list = np.concatenate(rt_file_list1)

# concat radiative transfer files
rt_files = []
for file in rt_file_list[0:2]:
    rt_files.append(xr.open_dataset(file))

rt_data = xr.concat(rt_files, dim='n_date')

# set frequency and date as coordinates
rt_data = rt_data.set_coords(['frequency', 'date'])

# run the makeRetrieval class

if ret_type == 'tpb':
    results = []
    for ret_spec in ret_specs:
        #print(ret_spec)
        ret_specs[ret_spec].update(params)
        results.append(MakeRetrieval(ret_specs[ret_spec],
                                     rt_data,
                                     ret_type,
                                     )
                       )
else:
    MakeRetrieval(params,
                  rt_data,
                  ret_type,
                  )