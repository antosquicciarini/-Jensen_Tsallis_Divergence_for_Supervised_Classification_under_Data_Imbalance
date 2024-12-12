from __future__ import print_function
import argparse
import datetime
import pandas as pd
import re

cf_dict = {
    "jensen_tsallis_loss": "JT",
    "jensen_shannon_loss": "JS",
    "jensen_reyni_loss": "JR",
    "cross_entropy_loss": "CE",
    "cross_entropy_ER_loss": "CR",
    "BrierLoss": "BL",
    "MeanAbsoluteError": "MAE",
    "focal_loss": "FL"
}


def upload_cf_params(args):
    
    if args.dataset == "MNIST":
        df = pd.read_csv('programs/best_parameters/best_parameters__MNIST.csv')
    elif args.dataset == "MNIST_fashion":
        df = pd.read_csv('programs/best_parameters/best_parameters__fashion_MNIST.csv')
    elif args.dataset == "CIFAR10":
        df = pd.read_csv('programs/best_parameters/best_parameters__CIFAR10.csv')

        
    pattern = r'[-+]?[0-9]*\.?[0-9]+'
    # Function to convert string to list of floats
    def string_to_float_list(s):
        matches = re.findall(pattern, s)
        return [float(match) for match in matches]
    
    df['par'] = df['par'].apply(string_to_float_list)

    if not hasattr(args, 'data_imbalance_mu'):
        args.data_imbalance_mu = 0.0
    if not hasattr(args, 'data_imbalance_rho'):
        args.data_imbalance_rho = 0.0

    cf_par = df.query(f"imb_rho == {args.data_imbalance_rho} and imb_mu == {args.data_imbalance_mu} and lf == '{cf_dict[args.loss_function]}'")['par']

    if cf_dict[args.loss_function] == "JT":
        args.pi_loss = cf_par.iloc[0][0]
        args.q_loss = cf_par.iloc[0][1]

    elif cf_dict[args.loss_function] == "JS":
        args.pi_loss = cf_par.iloc[0][0]

    elif cf_dict[args.loss_function] == "FL":
        args.gamma_focal_loss = cf_par.iloc[0][0]

    print("UPLOADED PARAMETERS: \n mu = {} \n rho = {} \n CF - {} \n par {}".format(args.data_imbalance_mu, args.data_imbalance_rho, args.loss_function, cf_par))
    return args


def parameter_setting(params, job_name):
    parser = argparse.ArgumentParser(description='JTD_imbalance')

    # Job and experiment settings
    parser.add_argument('--job-name', type=str, default=job_name, help='set job name')
    parser.add_argument('--experiment-name', type=str, default=params.get('experiment_name', 'general_experiment'), help='set experiment name')
    parser.add_argument('--dataset', type=str, default=params['dataset'], help='set dataset')

    # Data split settings
    if 'train_valid_split' in params:
        parser.add_argument('--train-valid-split', type=float, default=params['train_valid_split'], help='train valid split')
    
    parser.add_argument('--fast-training', action='store_true', default=params.get('fast_training', False), help='reduce operations to improve execution speed')

    # Data imbalance settings
    if params.get('data_imbalance', False):
        parser.add_argument('--data-imbalance', action='store_true', default=params['data_imbalance'], help='data imbalance flag')
        parser.add_argument('--data-imbalance-type', type=str, default=params['data_imbalance_type'], help='data imbalance type')
        parser.add_argument('--data-imbalance-rho', type=int, default=params['data_imbalance_rho'], help='data imbalance rho')
        
        if params['data_imbalance_type'] == "step":
            parser.add_argument('--data-imbalance-mu', type=int, default=params['data_imbalance_mu'], help='data imbalance mu')
        
        parser.add_argument('--data-imbalance-ROS', action='store_true', default=params['data_imbalance_ROS'], help='data imbalance ROS')

    # Network parameters
    parser.add_argument('--network', type=str, default=params.get('network'), help='define network structure')

    # Optimization parameters
    parser.add_argument('--batch-size', type=int, default=params.get('batch_size', 64), metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=params.get('test_batch_size', 1000), metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=params.get('epochs', 14), metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--optimizer', type=str, default=params.get('optimizer', 'Adadelta'), help='set optimizer (default: Adadelta)')
    parser.add_argument('--loss-function', type=str, default=params.get('loss_function', 'Shannon_cross_entropy'), help='set loss function (default: Shannon_cross_entropy)')

    if 'jensen' in params['loss_function']:
        parser.add_argument('--pi-loss', type=float, default=params.get('pi_loss', 0.1), help='Pi Jensen Coefficient')
        
        if 'tsallis' in params['loss_function']:
            parser.add_argument('--q-loss', type=float, default=params.get('q_loss', 1.5), help='Tsallis Coefficient')

    if 'alpha_loss' in params:
        parser.add_argument('--alpha-loss', type=float, default=params['alpha_loss'], help='Alpha Renyi Coefficient')

    if 'gamma_focal_loss' in params:
        parser.add_argument('--gamma-focal-loss', type=float, default=params['gamma_focal_loss'], help='gamma focal loss')
        
    parser.add_argument('--lr', type=float, default=params.get('lr', 1.0), metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=params.get('momentum', 0.9), help='momentum (default: 0.9)')
    parser.add_argument('--nesterov', type=bool, default=params.get('nesterov', False), help='nesterov (default: False)')
    parser.add_argument('--weight-decay', type=float, default=params.get('weight_decay', 0.0), help='weight decay (default: 0.0)')

    # Early stopping
    parser.add_argument('--early-stopping', action='store_true', default=params.get('early_stopping', False), help='activate early stopping')
    if params.get('early_stopping', False):
        parser.add_argument('--early-stopping-patience', type=int, default=params.get('early_stopping_patience'), help='set patience')
        parser.add_argument('--early-stopping-observed-value', type=str, default=params.get('early_stopping_observed_value', "loss"), help='set observed value for early stopping')

    # Learning rate policy
    parser.add_argument('--lr-policy', type=str, default=params.get('lr_policy'), help='set learning rate policy')
    parser.add_argument('--lr-step-reduction-rate', type=float, default=params.get('lr_step_reduction_rate'), help='set learning rate step reduction rate')
    if params.get('lr_policy') == 'lr_step':
        parser.add_argument('--lr-step-max-n-jumps', type=int, default=params.get('lr_step_max_n_jumps'), help='set maximum number of jumps for lr step')
        parser.add_argument('--lr-step-patience', type=int, default=params.get('lr_step_patience'), help='set patience for lr step')
        parser.add_argument('--lr-step-observed-value', type=str, default=params.get('lr_step_observed_value', "loss"), help='set observed value for lr step')
    elif params.get('lr_policy') == 'lr_milestones':
        parser.add_argument('--lr-milestones', type=list, default=params['lr_milestones'], help='set milestones for learning rate')

    parser.add_argument('--lr-warm-up', action='store_true', default=params.get('lr_warm_up', False), help='enable learning rate warm up')
    if params.get('lr_warm_up', False):
        parser.add_argument('--lr-warm-up-epochs', type=int, default=params.get('lr_warm_up_epochs', 10), help='set number of warm up epochs')

    # Computational settings
    parser.add_argument('--no-cuda', action='store_true', default=params.get('no_cuda', False), help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=params.get('no_mps', False), help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=params.get('dry_run', False), help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=params.get('seed', 18), metavar='S', help='random seed (default: 18)')
    parser.add_argument('--log-interval', type=int, default=params.get('log_interval', 10), metavar='N', help='number of batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=params.get('save_model', False), help='save the current model')
    parser.add_argument('--upload-best-parameters', action='store_true', default=params.get('upload_best_parameters', False), help='upload best parameters')

    args = parser.parse_args()

    if args.upload_best_parameters:
        args = upload_cf_params(args)
        
    print("Model Settings:")
    for key, value in vars(args).items():
        print(f"{key} --> {value}")
        
    now = datetime.datetime.now()
    args.model_name = f'{args.dataset}_{now.strftime("%m_%d_%H_%M_%S")}'
    print(args.dataset)
    return args