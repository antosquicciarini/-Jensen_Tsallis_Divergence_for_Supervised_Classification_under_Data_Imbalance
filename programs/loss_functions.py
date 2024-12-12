from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

epsilon = 1e-15

def sum_dim_ind(data):
    if len(data.shape)>1:
        return torch.sum(data, dim=1)
    else:
        return torch.sum(data)

def log_q(x, q):
    return (x**(1-q) - 1) / (1-q)

def relative_entropy(p_1, p_2, q=1.0, alpha=1.0):
    p_corr_1 = p_1 + epsilon
    p_corr_2 = p_2 + epsilon

    if q==1.0 and alpha==1.0:
        return (p_corr_1*torch.log(p_corr_1/p_corr_2)).sum(axis=1) #Kullback Leibler
    elif q!=1.0 and alpha == 1.0: 
        return (-p_corr_1*log_q(p_corr_2/p_corr_1, q)).sum(axis=1) #Tsallis Relative entropy
    elif q==1.0 and alpha != 1.0: 
        return ((1/(alpha-1))*torch.log((p_corr_1**alpha * p_corr_2**(1-alpha)).sum(axis=1))) #Reyni Relative entropy


class jensen_loss(nn.Module):
    def __init__(self, args):
        super(jensen_loss, self).__init__()

        self.args = args
        self.rescale_flag = args.jensen_shannon_loss_rescaled

        if hasattr(args, 'q_loss'):
            self.q_loss = args.q_loss
        else: 
            self.q_loss = 1.0

        if hasattr(args, 'alpha_loss'):
            self.alpha_loss = args.alpha_loss
        else: 
            self.alpha_loss = 1.0
            
        if hasattr(args, 'pi_loss'):
            self.pi_loss = args.pi_loss
        else: 
            self.pi_loss = 0.5

    def forward(self, logits, targets, reduction = "mean"):
    
        targets_cat = F.one_hot(torch.tensor(targets), num_classes=self.args.num_classes).float()
        predictions = F.softmax(logits, dim=1)
        if len(predictions.shape)==2:
            concatenated_tensor = torch.cat((targets_cat.unsqueeze(-1), predictions.unsqueeze(-1)), dim=-1)
        elif len(predictions.shape)>2:
            concatenated_tensor = torch.cat((targets_cat.unsqueeze(-1), predictions), dim=-1)

        pi_coef = torch.cat((torch.tensor([self.pi_loss]), (1-self.pi_loss)/(len(predictions.shape)-1) * torch.ones(len(predictions.shape)-1)), dim=0)
        pi_coef = pi_coef.to(concatenated_tensor.device)

        scaled_concatenated_tensor = concatenated_tensor * pi_coef.view(1, 1, -1)
        mean_dist = scaled_concatenated_tensor.sum(axis=2)

        loss = 0
        for ii in range(len(predictions.shape)):
            p_i = concatenated_tensor[:,:,ii]
            loss += pi_coef[ii] * relative_entropy(p_i, mean_dist, q = self.q_loss, alpha=self.alpha_loss)

        if reduction == "mean":
            return torch.mean(loss)
        elif reduction == "sum":
            return torch.sum(loss)
        elif reduction == "none":
            return loss
        

class focal_loss(nn.Module):
    def __init__(self, args):
        super(focal_loss, self).__init__()
        self.args = args
        self.gamma = args.gamma_focal_loss

    def forward(self, logits, targets, reduction="mean"):
        targets_cat = F.one_hot(torch.tensor(targets), num_classes=self.args.num_classes).float()
        confidences = F.softmax(logits, dim=1)
        confidences_correct_class = (confidences*targets_cat).sum(dim=1).clamp(min=1e-15)
        loss_score = - confidences_correct_class.log() * (1-confidences_correct_class)**self.gamma

        if reduction == "mean":
            return torch.mean(loss_score)
        elif reduction == "sum":
            return torch.sum(loss_score)
        elif reduction == "none":
            return loss_score
        

class cross_entropy_loss(nn.Module):
    def __init__(self, args):
        super(cross_entropy_loss, self).__init__()
        self.args = args
    def forward(self, logits, targets, reduction="mean"):
        targets_cat = F.one_hot(torch.tensor(targets), num_classes=self.args.num_classes).float()
        return F.cross_entropy(logits, targets_cat, reduction=reduction) #"sum", "mean", "none"


class BrierLoss(nn.Module):
    def __init__(self, args):
        super(BrierLoss, self).__init__()
        self.args = args

    def forward(self, logits, targets, reduction = "mean"):
        targets_cat = F.one_hot(torch.tensor(targets), num_classes=self.args.num_classes).float()
        predictions = F.softmax(logits, dim=1)
        loss = torch.sum((predictions - targets_cat) ** 2, dim=1)

        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        return loss
    

class relative_entropy_loss(nn.Module):
    def __init__(self, args):
        super(relative_entropy_loss, self).__init__()

        self.args = args

        if hasattr(args, 'q_loss'):
            self.q_loss = args.q_loss
        else: 
            self.q_loss = 1.0

        if hasattr(args, 'alpha_loss'):
            self.alpha_loss = args.alpha_loss
        else: 
            self.alpha_loss = 1.0
    
    def forward(self, logits, targets, reduction = "mean"):
        targets_cat = F.one_hot(torch.tensor(targets), num_classes=self.args.num_classes).float()
        predictions = F.softmax(logits, dim=1)
        
        loss = relative_entropy(targets_cat, predictions, q = self.q_loss, alpha = self.alpha_loss)

        if reduction == "mean":
            return torch.mean(loss)
        elif reduction == "sum":
            return torch.sum(loss)
        elif reduction == "none":
            return loss
        

class MeanAbsoluteError(nn.Module):
    def __init__(self, args):
        super(MeanAbsoluteError, self).__init__()
        self.args = args

    def forward(self, logits, targets, reduction = "mean"):
        targets_cat = F.one_hot(torch.tensor(targets), num_classes=self.args.num_classes).float()
        predictions = F.softmax(logits, dim=1)
        loss = 2 - 2*torch.sum(targets_cat*predictions, dim=1)

        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss

        return loss
    

def return_loss_function(args):
        
    if "jensen" in args.loss_function:
        loss_function = jensen_loss(args)

    elif "cross_entropy" in args.loss_function:
        loss_function = cross_entropy_loss(args)

    elif "BrierLoss" in args.loss_function:
        loss_function = BrierLoss(args)

    elif "MeanAbsoluteError" in args.loss_function:
        loss_function = MeanAbsoluteError(args)

    elif 'focal_loss' in args.loss_function:
        loss_function = focal_loss(args)

    return loss_function, args