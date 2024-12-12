from __future__ import print_function
import pandas as pd
import numpy as np
from train_network import train, test


def network_epoch_iteration(model, device, train_loader, test_loader, optimizer, loss_function, args):

    # Initialize variables
    if args.early_stopping:
        best_early_stop = float('inf')
    if args.lr_policy == "lr_step":
        best_lr_step = float('inf')
        lr_step_counter = 0

    if args.lr_warm_up:
        lr_warm_up_flag = True
        lr_warm_up_counter = 0
    else:
        lr_warm_up_flag = False

    #Initialize list
    train_result_dict_list = []
    test_result_dict_list = []

    for epoch in range(1, args.epochs + 1):

        if lr_warm_up_flag:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * epoch/args.lr_warm_up_epochs
            print(f"Warm up: lr={args.lr * epoch/args.lr_warm_up_epochs}")
            lr_warm_up_counter += 1
            if lr_warm_up_counter == args.lr_warm_up_epochs:
                lr_warm_up_flag = False
                param_group['lr'] = args.lr
                print(f"Warm up competed: lr={args.lr}")
    
        train_result_dict = train(args, model, device, train_loader, optimizer, epoch, loss_function)
        test_result_dict = test(args, model, device, test_loader, epoch, loss_function)
        
        train_result_dict_list.append(train_result_dict)
        test_result_dict_list.append(test_result_dict)

        if args.early_stopping:
            if test_result_dict[args.early_stopping_observed_value] < best_early_stop:
                best_early_stop = test_result_dict[args.early_stopping_observed_value]


                # Save the model's state_dict
                best_model_state_dict = model.state_dict()  
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.early_stopping_patience:
                    print(f"Early stopping after {epoch} epochs without improvement.")
                    break
            print(f"Early stopping counter {early_stopping_counter}/{args.early_stopping_patience}")

        else:
            best_model_state_dict = model.state_dict()

        if not lr_warm_up_flag: #The warm up desactivates other lr policies
            if args.lr_policy == "lr_step":
                if test_result_dict[args.lr_step_observed_value] < best_lr_step:
                    best_lr_step = test_result_dict[args.lr_step_observed_value]
                    lr_patience_counter = 0
                else:
                    lr_patience_counter += 1
                    if lr_patience_counter >= args.lr_step_patience:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= args.lr_step_reduction_rate
                        print(f"Reducing learning rate to {param_group['lr']}")
                        lr_patience_counter = 0
                        lr_step_counter += 1
                        if lr_step_counter>=args.lr_step_max_n_jumps:
                            print(f"Maximum number of learning rate jumps achieved after {epoch} epochs.")
                            break
                print(f"lr patience {lr_patience_counter}/{args.lr_step_patience}")
                print(f"lr n° of steps {lr_step_counter}/{args.lr_step_max_n_jumps}")   

            elif args.lr_policy == "lr_milestones":
                if epoch in args.lr_milestones:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= args.lr_step_reduction_rate
                    print(f"Reducing learning rate to {param_group['lr']}")

            elif args.lr_policy == "lr_cos":
                lr_epoch = 0.5 * args.lr * (1 + np.cos(np.pi * epoch/args.epochs))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_epoch
                print(f"Reducing learning rate to {param_group['lr']}")   

    history_train = pd.DataFrame(train_result_dict_list)
    history_test = pd.DataFrame(test_result_dict_list)

    history_train['Epoch'] = range(1, len(history_train)+1)
    history_test['Epoch'] = range(1, len(history_test)+1)

    history_train['Data_Type'] = 'Train'
    history_test['Data_Type'] = 'Test'
    # Concatenate the DataFrames vertically to create a single DataFrame
    history = pd.concat([history_train, history_test], axis=0)
    # Reset the index of the combined DataFrame
    history.reset_index(drop=True, inplace=True)

    return model, history, best_model_state_dict