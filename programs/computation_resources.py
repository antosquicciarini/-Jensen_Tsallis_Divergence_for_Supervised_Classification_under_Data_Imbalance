from __future__ import print_function
import torch

def computation_resources(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        num_gpus = torch.cuda.device_count()
        print(f'n GPUs found: {num_gpus}')
        torch.cuda.set_device(0)
        device = torch.device("cuda")
    else:
        use_mps = not args.no_mps and torch.backends.mps.is_available()
        if use_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    else:
        mps_kwargs = {'shuffle': True}
        train_kwargs.update(mps_kwargs)
        test_kwargs.update(mps_kwargs)
        
    return device, train_kwargs, test_kwargs
