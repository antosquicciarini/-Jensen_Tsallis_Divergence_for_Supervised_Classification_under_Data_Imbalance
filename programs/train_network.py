from __future__ import print_function
import torch
import torch.nn.functional as F
import time
import torch

def compute_weight_norm(model):
    weight_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            weight_norm += torch.norm(param, p='fro')
    return weight_norm.item()


def model_evaluate(model, loader, device, loss_function, n_data, args, lr=None, test_flag=False):
    model.eval()

    data_limit = args.test_data  
    output_tot, target_tot, loss_tot = [], [], []

    softmax_output_tot, entropy = None, None

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            
            output_tot.append(output)
            target_tot.append(target)

            batch_loss = loss_function(output, target, reduction='none')
            loss_tot.append(batch_loss)

            if batch_idx*args.batch_size >= data_limit:
                n_data = args.batch_size * (batch_idx + 1)
                break

        target_tot = torch.cat(target_tot, dim=0).cpu().detach() #See if this is the problem
        output_tot = torch.cat(output_tot, dim=0).cpu().detach()
        loss_tot = torch.cat(loss_tot, dim=0)

        loss = loss_tot.sum().item() / n_data

        softmax_output_tot = F.softmax(output_tot, dim=1)
        entropy = -torch.sum(softmax_output_tot * torch.log(softmax_output_tot + 1e-7), dim=1)

    confidences_top_5, prediction_top_5 = softmax_output_tot.topk(5, 1, largest=True, sorted=True)
    target_top_5 = target_tot.view(target_tot.size(0), -1).expand_as(prediction_top_5)
    correct_top_5 = prediction_top_5.eq(target_top_5).float()

    correct = correct_top_5[:, :1].sum().item()
    correct_5 = correct_top_5.sum().item()

    accuracy = correct / n_data

    correct_mask = correct_top_5[:, :1].squeeze(dim=1).bool()
    incorrect = (~correct_mask).sum().item()
    
    correct_loss = loss_tot.sum().item() / n_data
    incorrect_loss = loss_tot.sum().item() / n_data

    if correct > 0:
        entropy_correct = entropy[correct_mask].sum().item() / correct
    elif correct == 0:
        entropy_correct = 0
    if incorrect>0.0:
        entropy_incorrect = entropy[~correct_mask].sum().item() / incorrect
    elif incorrect==0:
        entropy_incorrect = 0

    result_dict = {
        "loss": loss,
        "accuracy": accuracy,
        "accuracy_top_5": correct_5 / n_data,
        "error": 1 - accuracy,
        "correct": correct,
        "incorrect": incorrect,
        "correct_loss": correct_loss,
        "incorrect_loss": incorrect_loss,
        "entropy_correct": entropy_correct,
        "entropy_incorrect": entropy_incorrect,
    }

    if lr is not None:
        result_dict["lr"] = lr
        result_dict["weight_norm"] = compute_weight_norm(model)

    return result_dict


def train(args, model, device, train_loader, optimizer, epoch, loss_function):
    
    start_time = time.time()
    model.train()

    model, batch_idx = epoch_train(train_loader, device, optimizer, model, loss_function, epoch, args)
    end_time_epoch = time.time()

    print(f"Total Epoch Time: {end_time_epoch - start_time:.2f} seconds")
    print(f"Average Batch Time: {(end_time_epoch - start_time)/batch_idx:.2f} seconds")

    for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']
    
    if not args.fast_training or epoch == args.epochs:
        print("Train Evaluation...")
        train_result_dict = model_evaluate(model, train_loader, device, loss_function, args.training_data, args, lr=learning_rate)
        print_results(train_result_dict, args)
        print(f"Train Evaluation Time: {(time.time()-end_time_epoch):.2f} seconds")
    else: 
        train_result_dict = {}

    return train_result_dict


def epoch_train_generalised(train_loader, device, optimizer, model, loss_function, epoch, args):

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        target = target.to(device)
        output_list = []
        for i in range(args.M_generalised_JD):
            torch.manual_seed(i)
            augmented_data = torch.stack([args.DA_transform(data[i]) for i in range(data.size(0))])
            augmented_data = augmented_data.to(device)
            output_list.append(model(augmented_data))
        
        loss = loss_function(output_list, target)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0 and len(data)==args.batch_size:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    return model, batch_idx


def epoch_train(train_loader, device, optimizer, model, loss_function, epoch, args):
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0 and len(data)==args.batch_size:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

            
    return model, batch_idx


def test(args, model, device, test_loader, epoch, loss_function):

    if not args.fast_training or epoch == args.epochs:
        start_time = time.time()
        print("Test Evaluation...")
        test_result_dict = model_evaluate(model, test_loader, device, loss_function, args.test_data, args, test_flag=True)
        print_results(test_result_dict, args, type_of_data="Test")
        print(f"Test Evaluation Time: {(time.time()-start_time):.2f} seconds")
    else: 
        test_result_dict = {}

    return test_result_dict


def print_results(result_dict, args, type_of_data="Train"):
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        type_of_data, result_dict['loss'], result_dict['correct'], result_dict['correct']+result_dict['incorrect'], result_dict['accuracy']*100))
    print('Top-5 Accuracy: {:.0f}%'.format(result_dict['accuracy_top_5']*100))
    print('Correctly Classified Loss: {:.4f}'.format(result_dict['correct_loss']))
    print('Incorrectly Classified Loss: {:.4f}'.format(result_dict['incorrect_loss']))
    print('Entropy of Correct Outputs: {:.4f}'.format(result_dict['entropy_correct']))
    print('Entropy of Incorrect Outputs: {:.4f}'.format(result_dict['entropy_incorrect']))