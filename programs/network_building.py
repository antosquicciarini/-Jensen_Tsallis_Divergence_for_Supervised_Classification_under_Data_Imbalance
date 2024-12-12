from __future__ import print_function
import torch
import torch.optim as optim
import os
import json
import numpy as np
from loss_functions import return_loss_function
from torchvision.transforms import Compose
from network_structure import select_network
from network_epoch_iteration import network_epoch_iteration

from plots import plot_history_pro


def network_building(args, device, train_loader, test_loader):

    # For reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = select_network(args, device)

    # Print the network structure
    print(model)
    total_parameters = 0
    model_txt = ''
    for name, param in model.named_parameters():

        if "bias" not in name:
            print(f"{name} -- {param.shape} - {param.numel()}\n")
            model_txt += f"{name} -- {param.shape} - {param.numel()}\n"

        if param.requires_grad:
            total_parameters += param.numel()
            
    print(f"\nTotal Parameters -- {total_parameters}")
    model_txt += f"Total Parameters -- {total_parameters}"
    
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unsupported optimizer: {}".format(args.optimizer))
    
    loss_function, args = return_loss_function(args)

    model, history, best_model_state_dict = network_epoch_iteration(model, device, train_loader, test_loader, optimizer, loss_function, args)

    if args.save_model:

        args.experiment_folder = f'models/{args.experiment_name}'
        args.model_folder = f"models/{args.experiment_name}/{args.model_name}"

        def try_create_folder(path):
            if os.path.isdir(path):
                print("Path already exists: " + path)
            else:
                os.mkdir(path)
                print("Path created: " + path)


        try_create_folder(args.experiment_folder)
        try_create_folder(args.model_folder)

        torch.save(best_model_state_dict, f'{args.model_folder}/{args.model_name}.pt')
        
        history.to_csv(f'{args.model_folder}/{args.model_name}_history.csv', index=False)

        ### Save the arguments to a JSON file
        args_dict = vars(args)
        
        def convert_to_serializable(obj):
            if isinstance(obj, Compose):
                # Convert the Compose object to a list of dictionaries
                return [{'transform': str(transform)} for transform in obj.transforms]
            # Add more custom conversions as needed
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        if not args.fast_training:
            plot_history_pro(history, args)

        # Serialize with the custom conversion function
        with open(f'{args.model_folder}/{args.model_name}_args.json', 'w') as json_file:
            json.dump(args_dict, json_file, indent=4, default=convert_to_serializable)

        return None


