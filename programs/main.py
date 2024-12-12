from __future__ import print_function
import json
import time
import itertools
import os
import logging
import sys
import pdb

from parameter_setting import parameter_setting
from load_data import load_data
from network_building import network_building
from computation_resources import computation_resources

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_parameters_from_json(json_filename):
    with open(json_filename, 'r') as json_file:
        params = json.load(json_file)
    return params

def remove_duplicates(input_list):
    seen = set()
    unique_list = []
    for d in input_list:
        # Convert the dictionary to a frozenset to make it hashable
        frozen_dict = frozenset(d.items())
        # Check if the frozenset is in the set of seen items
        if frozen_dict not in seen:
            seen.add(frozen_dict)
            unique_list.append(d)
    return unique_list

def generate_settings_combinations(original_dict):
    # Create a list of keys with lists as values
    list_keys = [key for key, value in original_dict.items() if isinstance(value, list)]
    # Generate all possible combinations
    combinations = [list(comb) for comb in itertools.product(*[original_dict[key] for key in list_keys])]
    # Create a list of dictionaries with unique combinations
    result = []
    for combo in combinations:
        new_dict = original_dict.copy()
        for key, value in zip(list_keys, combo):
            new_dict[key] = value
        result.append(new_dict)

    for dict in result:

        if 'q_loss' in dict:
            if 'tsallis' not in dict['loss_function']:
                del dict['q_loss']

        if 'alpha_loss' in dict:
            if 'renyi' not in dict['loss_function']:
                del dict['alpha_loss']

        if 'pi_loss' in dict:
            if 'jensen' not in dict['loss_function']:
                del dict['pi_loss']

        if 'gamma_focal_loss' in dict:
            if 'focal_loss' not in dict['loss_function']:
                del dict['gamma_focal_loss']

        if not dict.get('data_imbalance', False):
            key_to_delete = []
            for k, v in dict.items():
                if 'data_imbalance_' in k: #keep data imbalance flag, deleting the other keys
                    key_to_delete.append(k)
            for k in key_to_delete:
                del dict[k]
        if dict.get('data_imbalance_type', None) == "linear":
            if 'data_imbalance_mu' in dict:
                del dict['data_imbalance_mu']

    result = remove_duplicates(result)

    return result


def calculate_elapsed_time(start_time, current_index, total_experiments):

    elapsed_time = time.time() - start_time
    days, remainder = divmod(elapsed_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"Time elapsed for current experiment: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")

    remaining_time = elapsed_time * (len(total_experiments) - current_index - 1)
    days, remainder = divmod(remaining_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"Estimated remaining time: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")

def main():

    # set cwd project's root 
    if "programs" in os.getcwd():
        os.chdir("..")

    logging.info("Start!")
    job_name = sys.argv[2]

    json_filename = f'programs/JSON_parameters/{job_name}.json'
    params_json = load_parameters_from_json(json_filename)
    params_list = generate_settings_combinations(params_json)
    logging.info(f"Total number of experiments: {len(params_list)}")

    for ii, params in enumerate(params_list):

        start_time = time.time()
        logging.info(f"Starting experiment {ii+1}/{len(params_list)} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        params['experiment_name'] = job_name
        args = parameter_setting(params, job_name)
        device, train_kwargs, test_kwargs = computation_resources(args)
        args, train_loader, test_loader = load_data(args, train_kwargs, test_kwargs)
        network_building(args, device, train_loader, test_loader)
        calculate_elapsed_time(start_time, ii, params_list)


if __name__ == '__main__':
    main()