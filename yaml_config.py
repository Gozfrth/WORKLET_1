import yaml


# model = YOLO("runs\\detect\\train\\weights\\best.pt")
# metrics = model.val()


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
                    data[key] = array[index]  # Update the value based on the index
                else:
                    update_values(value, config)  # Recurse if the value is nested
        elif isinstance(data, list):
            for item in data:
                update_values(item, config)

    update_values(data, config)

    with open(output_file, 'w') as file:
        yaml.dump(data, file, Dumper=NoSortDumper, default_flow_style=False)




config = {
    "gamma": 0,   # Index in the array for ... `key`?
    "intensity_sigma": 1,   # Index in the array for `key2`
    "bl_r": 2 ,                            # a subtractive value, not additive!
    "bl_gr": 3,
    "bl_gb": 4,
    "bl_b": 5,
    "alpha": 6,                            # x1024
    "beta": 7,                           # x1024
}

print("_______________________")
print(config)
print("_______________________")
array = [0.60, 0.9, 0.1, 0.2, 0.1, 0.01, 0.2, 0.3]  # Array of values to map

input_file = 'configs/test_emmetra.yaml'

output_file = 'configs/test_emmetra_copy.yaml'

modify_yaml(input_file, output_file, config, array)

temp_config = {"bl_r" : 0, "bl_gr" : 0}




### 


def extract_module_parameters(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    params = {}

    for module, parameters in config.items():
        params[module] = parameters
        
    return params

base_params = extract_module_parameters('configs/test_emmetra.yaml')

def write_module_parameters(file_path, params):
    for module, parameters in temp_config.items():
        base_params[module] = params[temp_config[module]]

    with open(file_path, 'w') as file:
        yaml.dump(base_params, file)
    

def extract_module_static_parameters(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    static_params = {}

    for module, parameters in config.items():
        if module in ['module_enable_status', 'hardware']:
            static_params[module] = parameters
        
    return static_params

def extract_module_static_parameters(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    flattened_params = []
    param_keys = []

    for module, parameters in config.items():
        if module in ['module_enable_status', 'hardware']:
            continue
        
        count = 0

        if parameters is not None:
            for key, value in parameters.items():
                if key in ['dpc', 'cnf', 'nlm', 'bnf', 'blc']:
                    if isinstance(value, (int, float)):
                        config[key] = count
                        count += 1
                        flattened_params.append(value)
                        param_keys.append(f"{module}.{key}")

    return flatten_list(flattened_params), param_keys

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