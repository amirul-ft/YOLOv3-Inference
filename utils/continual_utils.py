import time, random, copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import utils.utils

from utils.utils import torch_utils

device = torch_utils.select_device('0')

epsilon = 1e-7
si_c = 1e-5
SI_parameters = {
    'previous_task':{},
    'old_parameters':{},
    'omega':{},
    'W':{}
}

'--- Synaptic Intelligence (SI) Specific Functions ---'

def si_register_parameters(model):
    for n, p in model.named_parameters():
        pname = n.replace('.', '_')
        if 'YOLOLayer' in pname:
            continue
            
        SI_parameters['previous_task'][pname] = p.detach().clone().zero_()
        SI_parameters['old_parameters'][pname] = p.detach().clone().zero_()
        SI_parameters['omega'][pname] = p.detach().clone().zero_()
        SI_parameters['W'][pname] = p.detach().clone().zero_()

def si_update_omega(model):
    layers = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            layers.append(n)
            pname = n.replace('.', '_')
            if 'YOLO' in pname:
                continue
            
            p_prev = SI_parameters['previous_task'][pname]
            p_current = p.detach().clone()
            p_change = p_current - p_prev
            delta_omega = SI_parameters['W'][pname] / (p_change**2 + epsilon)
            omega = SI_parameters['omega'][pname]
            new_omega = omega + delta_omega

            SI_parameters['previous_task'][pname] = p_current
            SI_parameters['omega'][pname] = new_omega
            
            SI_parameters['W'][pname] = p.detach().clone().zero_()
    print(layers)

def si_update_W(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            pname = n.replace('.', '_')
            if 'YOLOLayer' in pname:
                continue
            
            if p.grad is not None:
                delta_W = -p.grad * (p.detach() - SI_parameters['old_parameters'][pname])
                SI_parameters['W'][pname].add_(delta_W)
            SI_parameters['old_parameters'][pname] = p.detach().clone()
            
def si_surrogate_loss(model):
    losses = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            pname = n.replace('.', '_')
            if 'YOLOLayer' in pname:
                continue
                
            prev_values = SI_parameters['previous_task'][pname]
            omega = SI_parameters['omega'][pname]
            losses.append((omega * (p - prev_values)**2).sum())
    return sum(losses)

    '--- Deep Model Consolidation (DMC) Specific Functions ---'

DMC_parameters = {
    'previous_task' : {},
    'old_parameters' : {},
    'omega' : {},
    'W' : {}
}

"""
def DMC_init_parameters(model):
    for n, p in model.named_parameters():
        pname = n.replace('.', '_')
        if 'YOLOLayer' in pname:
            continue
            
        DMC_parameters['previous_task'][pname] = p.detach().clone().zero_()
        DMC_parameters['old_parameters'][pname] = p.detach().clone().zero_()
        DMC_parameters['omega'][pname] = p.detach().clone().zero_()
        DMC_parameters['W'][pname] = p.detach().clone().zero_()

def DMC_update_parameters(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            pname = n.replace('.', '_')
"""

def double_L2_losses(current_logits, previous_logits, number_of_task):
    losses = ((current_logits - previous_logits)**2)/number_of_task
    return torch.sum(losses)

def smooth_bbox_losses(p, current_loss, model):
    utils.compute_loss(p, target, model)
    return 0

def consolidation(prev_model, current_model, data, number_of_task=2, epochs=100):
    for epoch in range(0, eopchs):
        prev_pred = first_model(data)
        current_pred = current_model(data)
        double_L2_losses(current_model.logits, prev_model.logits, number_of_task)




