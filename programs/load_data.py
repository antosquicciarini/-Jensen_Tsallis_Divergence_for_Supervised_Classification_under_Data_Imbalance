import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader
import random
from collections import defaultdict

# Define constants
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
FASHION_MNIST_MEAN, FASHION_MNIST_STD = (0.5,), (0.5,)
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def stratified_split(indices, targets, args):

    indices_per_label = defaultdict(list)
    
    for indx in indices:
        indices_per_label[targets[indx]].append(indx)
        
    first_set_indices, second_set_indices = list(), list()

    for label, indices_per_label in indices_per_label.items():
        n_samples_for_label = round(len(indices_per_label) * args.train_valid_split)
        random_indices_sample = random.sample(indices_per_label, n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices_per_label) - set(random_indices_sample))

    return first_set_indices, second_set_indices

def apply_data_imbalance(indices, targets, args):

    # Create a new dataset excluding instances with the specified indices
    if args.data_imbalance_type == "linear":
        args.reduction_rates = torch.linspace(1/args.data_imbalance_rho, 1, args.num_classes)

    elif args.data_imbalance_type == "step":
        n_classes_to_be_reduced = int(args.data_imbalance_mu*10)
        args.reduction_rates = torch.cat((1/args.data_imbalance_rho * torch.ones(n_classes_to_be_reduced), torch.ones(args.num_classes-n_classes_to_be_reduced)),dim=0)

    indices_data_imbalance = []

    indices_per_label = defaultdict(list)
    
    for indx in indices:
        indices_per_label[targets[indx]].append(indx)

    for label, indices_per_label in indices_per_label.items():

        num_samples_to_include = int(args.reduction_rates[label] * len(indices_per_label))
        if num_samples_to_include == 0:
            num_samples_to_include = 1 # at least one sample per class

        indices_data_imbalance.extend(np.random.choice(indices_per_label, num_samples_to_include, replace=False))

    args.reduction_rates = tuple(float(x) for x in args.reduction_rates) #To make it seriazible for Json

    return indices_data_imbalance, args
 
def load_data(args, train_kwargs, test_kwargs, transformer=None):

    # For reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #%% DATASET LOADING
    if args.dataset == "MNIST":
        MEAN, STD = MNIST_MEAN, MNIST_STD

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST('data', train=False, transform=transform)

    elif args.dataset == "MNIST_fashion":
        MEAN, STD = FASHION_MNIST_MEAN, FASHION_MNIST_STD

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(3), 
            transforms.ToTensor(), 
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        transform_test = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(), 
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        
        dataset1 = datasets.FashionMNIST('data', train=True, download=True, transform=transform_train)
        dataset2 = datasets.FashionMNIST('data', train=False, transform=transform_test)


    elif args.dataset == "CIFAR10":
        MEAN, STD = CIFAR10_MEAN, CIFAR10_STD

        if args.data_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        dataset1 = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
        dataset2 = datasets.CIFAR10('data', train=False, transform=transform_test)

    # Substitute pre-defined transformer
    if transformer is not None:
        dataset1.transform = transformer
        dataset2.transform = transformer

    args.num_classes = len(dataset1.classes)


    #%% SUB-DATASET Selection 
    indices = np.arange(len(dataset1))
    targets = dataset1.targets
    
    if not isinstance(targets, list):
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()

    # Train-Valid Split
    if hasattr(args, 'train_valid_split'):
        valid_set_indices, indices = stratified_split(indices, targets, args)

    # Class Imbalance
    if getattr(args, 'data_imbalance', False):
        print("Reducing dataset...")
        indices, args = apply_data_imbalance(indices, targets, args)

    # CHECK: train and valid set do not share any elements
    if hasattr(args, 'train_valid_split'):
        if set(indices).isdisjoint(set(valid_set_indices)):
            print("Train and Valid dataset do not share any elements")
        else:
            raise ValueError("Train and Valid dataset SHARE at least one element!!!")

    # Creating train/valid datasets
    train_dataset = torch.utils.data.Subset(dataset1, indices)

    if hasattr(args, 'train_valid_split'):
        valid_dataset = torch.utils.data.Subset(dataset1, valid_set_indices)
    else: 
        indices_test = np.arange(len(dataset2))
        valid_dataset = torch.utils.data.Subset(dataset2, indices_test) 

    if getattr(args, "data_imbalance_ROS", False):

        indices_per_label = defaultdict(list)
        for indx in indices:
            indices_per_label[targets[indx]].append(indx)
        
        label_counts = defaultdict(int)
        for label, indices_per_label in indices_per_label.items():
            label_counts[label] = len(indices_per_label)

        label_counts = np.array(sorted(label_counts.items()))
        weights = 1/label_counts[:,1]
        samples_weights = [weights[targets[indx]] for indx  in indices]
        sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights, 
                                                         num_samples=len(train_dataset),
                                                         replacement=True) 
        #Without True replacement it would not be able to replace at all
        del train_kwargs['shuffle']
        train_loader = DataLoader(train_dataset, sampler=sampler, **train_kwargs)
    else:
        train_loader = DataLoader(train_dataset, **train_kwargs)
        
    test_loader = DataLoader(valid_dataset, **test_kwargs)

    for data, target in train_loader:
       args.n_channels = data.shape[1]
       break
        
    args.training_data = len(train_dataset)
    args.test_data = len(valid_dataset)
    args.norm_par = (MEAN, STD)

    return args, train_loader, test_loader

