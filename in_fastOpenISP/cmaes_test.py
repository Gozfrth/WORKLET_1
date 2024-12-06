import numpy as np
import cv2

import time
import os
import yaml

from cmaes import CMA
from ultralytics import YOLO
from utils.yacs import Config
from pipeline import Pipeline


# model = YOLO("runs\\detect\\train\\weights\\best.pt")
# metrics = model.val()

# base_params = extract_module_parameters('configs/test_emmetra.yaml')
config = {}

selected_keys = [
    "gamma",
    "diff_threshold",
    "bl_r",
    "bl_gr",
    "bl_gb",
    "bl_b",
    "alpha",
    "beta",
    "r_gain",
    "gr_gain",
    "gb_gain",
    "b_gain",
    "intensity_sigma",
    "spatial_sigma",
]

class NoSortDumper(yaml.Dumper):
    def represent_dict(self, data):
        return self.represent_mapping('tag:yaml.org,2002:map', data.items())

NoSortDumper.add_representer(dict, NoSortDumper.represent_dict)

def represent_list(self, data):
    return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

NoSortDumper.add_representer(list, represent_list)

def modify_yaml(input_file, output_file, config, array):
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)
    
    def update_values(data, config):
        if isinstance(data, dict):
            for key, value in data.items():
                if key in config:
                    index = config[key]
                    data[key] = array[index] 
                else:
                    update_values(value, config) 
        elif isinstance(data, list):
            for item in data:
                update_values(item, config) 

    update_values(data, config)

    with open(output_file, 'w') as file:
        yaml.dump(data, file, Dumper=NoSortDumper, default_flow_style=False)    


def extract_module_parameters(file_path):
    with open(file_path, 'r') as file:
        cfg = yaml.safe_load(file)

    flattened_params = []
    param_keys = []

    count = 0

    # Iterate over each module in the configuration (cfg)
    for module, parameters in cfg.items():
        # Skip specific modules
        if module in ['module_enable_status', 'hardware']:
            continue

        # Make sure that parameters is not None
        if parameters is not None:
            for key, value in parameters.items():
                if isinstance(value, (int, float)):
                    flattened_params.append(value)  # Add value to flattened params list
                    param_keys.append(f"{module}.{key}")  # Create key path and add to list
                    count += 1

    return flattened_params, param_keys

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            # Recursively flatten nested lists
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_ITERS = 2
SIGMA = 1.3

def isp_pipeline(file_name):

    cfg = Config(file_name)
    # print(cfg)
    pipeline = Pipeline(cfg)

    raw_path = 'raw/rawFile.raw'
    bayer = np.fromfile(raw_path, dtype='uint16', sep='')
    bayer = bayer.reshape((cfg.hardware.raw_height, cfg.hardware.raw_width))

    data, _ = pipeline.execute(bayer)

    output_path = os.path.join(OUTPUT_DIR, 'test.png')
    output = cv2.cvtColor(data['output'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output)

    return data


def cmaes_iter(file_name):
    time.sleep(1)

    # ISP PIPELINE STEP HERE

    processed_image = isp_pipeline(file_name)

    # MODEL HERE
    # eval_metrics = model_eval(processed_image)


    return np.random.rand() # should return eval_metrics

def gen_file_name(generation, i, fields, x):

    file_name = f'configs/test_emmetra_gen_{generation}_iter_{i}.yaml'

    return file_name

if __name__ == "__main__":

    file_path = 'configs/test_emmetra.yaml'
    flattened_params, param_keys = extract_module_parameters(file_path)
    print(flattened_params)
    optimizer = CMA(mean=np.array(flattened_params), sigma=SIGMA)

    fields = param_keys

    for generation in range(MAX_ITERS):
        solutions = []

        for i in range(optimizer.population_size):
            x = optimizer.ask() # x - params
            print(x)
            
            generated_file_name = gen_file_name(generation, i, fields, x)
            modify_yaml(file_path, generated_file_name, config, x)
            
            print(f"\n\niter --- {generation} __ {i} \n\n")
            value = cmaes_iter(generated_file_name)
            solutions.append((x, value))
            
            print(f"Generation {generation}: value={value} params={x}")
        
        optimizer.tell(solutions)

# import numpy as np
# from cmaes import CMA, get_warm_start_mgd

# def source_task(x1: float, x2: float) -> float:
#     b = 0.4
#     return (x1 - b) ** 2 + (x2 - b) ** 2

# def target_task(x1: float, x2: float) -> float:
#     b = 0.6
#     return (x1 - b) ** 2 + (x2 - b) ** 2

# if __name__ == "__main__":
#     # Generate solutions from a source task
#     source_solutions = []
#     for _ in range(1000):
#         x = np.random.random(2)
#         value = source_task(x[0], x[1])
#         source_solutions.append((x, value))

#     # Estimate a promising distribution of the source task,
#     # then generate parameters of the multivariate gaussian distribution.
#     ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(
#         source_solutions, gamma=0.1, alpha=0.1
#     )
#     optimizer = CMA(mean=ws_mean, sigma=ws_sigma, cov=ws_cov)

#     # Run WS-CMA-ES
#     print(" g    f(x1,x2)     x1      x2  ")
#     print("===  ==========  ======  ======")
#     while True:
#         solutions = []
#         for _ in range(optimizer.population_size):
#             x = optimizer.ask()
#             value = target_task(x[0], x[1])
#             solutions.append((x, value))
#             print(
#                 f"{optimizer.generation:3d}  {value:10.5f}"
#                 f"  {x[0]:6.2f}  {x[1]:6.2f}"
#             )
#         optimizer.tell(solutions)

#         if optimizer.should_stop():
#             break