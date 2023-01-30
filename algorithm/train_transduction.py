import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from utils import move_data_to_device, bce, save_checkpoint
from evaluate_main import compare_single_chords

def train_transduction(training_config, transduction_nn, train_loader,
    validate_loader = None, save = True, plot_validate = False, plot_train = False
):
    print_iter = training_config["print_iter"]
    device = training_config["device"]
    torch_device = torch.device(device)
    
    if "cuda" in device:
        torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("training using cuda")
        transduction_nn.to(torch_device)
    
    optimizer = optim.Adam(transduction_nn.parameters(), 
        lr=training_config["learning_rate"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True
    )
    
    i_iter = 0
    if "checkpoint_name" in training_config:
        checkpoint_name = training_config["checkpoint_name"]
        checkpoint_path = os.path.join(training_config["checkpoint_dir"], checkpoint_name)
        if os.path.exists(checkpoint_path):
            i_iter = int(checkpoint_name.split("_")[1])+1
            training_config["start_train_iter"] = i_iter
            transduction_nn.load_state_dict(torch.load(checkpoint_path))
            print("loaded checkpoint ", checkpoint_name)
    else:
        training_config["start_train_iter"] = 0
    
    running_loss = 0.0
    running_TP = 0
    running_FP = 0
    running_FN = 0
    for i_iter in range(training_config["start_train_iter"], training_config["max_train_iter"]):
        torch.cuda.empty_cache()
        transduction_nn.train()
        optimizer.zero_grad()

        # forward data and calculate loss and backward
        batch_data_dict = next(iter(train_loader))
        est_velocity_rolls =  move_data_to_device(batch_data_dict["est_velocity_rolls"], torch_device) # (B,T,88)
        gt_velocity_rolls =  move_data_to_device((batch_data_dict["gt_velocity_rolls"]>0)*1.0, torch_device) # (B,T,88)
        durs =  torch.unsqueeze(move_data_to_device(batch_data_dict["offsets"] - batch_data_dict["onsets"], torch_device), 2) # (B,T,1)
        inputs = torch.cat([est_velocity_rolls, durs], dim = 2) # (B,T,89)
        
        smoothed_velocity_rolls = transduction_nn(inputs)
        
        mask = (gt_velocity_rolls * 0) + 1
        loss = 10*bce(smoothed_velocity_rolls, gt_velocity_rolls, mask)
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transduction_nn.parameters(), 0.01)
            running_loss += loss.item()
            optimizer.step()
        else:
            print("iter", i_iter, ", nan loss encountered...")

        # save checkpoint
        if save and i_iter % training_config["save_checkpoint_iter"] == 0:
            save_checkpoint(transduction_nn, training_config["checkpoint_dir"], i_iter, 
                model_name = training_config["model_name"]
            )

        # validate
        if (validate_loader is not None) and i_iter % print_iter == 0:
            with torch.no_grad():
                for _ in range(training_config["validate_iter"]):
                    batch_data_dict_val = next(iter(validate_loader))
                    transduction_nn.eval()
                    est_velocity_rolls_val =  move_data_to_device(batch_data_dict_val["est_velocity_rolls"], torch_device) # (B,T,88)
                    gt_velocity_rolls_val =  move_data_to_device((batch_data_dict_val["gt_velocity_rolls"]>0)*1.0, torch_device) # (B,T,88)
                    durs_val =  torch.unsqueeze(move_data_to_device(batch_data_dict_val["offsets"] - batch_data_dict_val["onsets"], torch_device), 2) # (B,T,1)
                    inputs_val = torch.cat([est_velocity_rolls_val, durs_val], dim = 2) # (B,T,89)
                    
                    smoothed_velocity_rolls_val = transduction_nn(inputs_val)
                    n_batch_val = smoothed_velocity_rolls_val.shape[0]
                    for i_batch in range(n_batch_val):
                        est_thres = 0.5
                        TP,FP,FN = compare_single_chords(
                            (gt_velocity_rolls_val[i_batch].cpu().numpy() > 0)*1, 
                            (smoothed_velocity_rolls_val[i_batch].cpu().numpy() > est_thres)*1
                        )
                        running_TP += TP
                        running_FP += FP
                        running_FN += FN
                        
                if plot_validate and i_iter % (print_iter*100) == 0:
                    fig, ax = plt.subplots(1,1,figsize = (12,4))
                    ax.plot(gt_velocity_rolls_val[0,0,:].cpu().numpy())
                    ax.plot(smoothed_velocity_rolls_val[0,0,:].cpu().numpy())
                    ax.set_title("validate result at iter "+str(i_iter))
                    ax.legend(["ground-truth", "estimated"])

        # print statistics
        if i_iter % print_iter == 0:
            if running_TP > 0:
                eval_P = 1.0*running_TP / (running_TP + running_FP)
                eval_R = 1.0*running_TP / (running_TP + running_FN)
                eval_F1 = 2*eval_P*eval_R / (eval_P + eval_R)
            else:
                eval_P = 0
                eval_R = 0
                eval_F1 = 0
            
            if validate_loader is not None:
                print(f'[{i_iter:5d}] loss: {running_loss/print_iter:.5f}; validate P,R,F1: {eval_P:.4f},{eval_R:.4f},{eval_F1:.4f}')
            else:
                print(f'[{i_iter:5d}] loss: {running_loss/print_iter:.5f}')
            running_loss = 0
            running_TP = 0
            running_FP = 0
            running_FN = 0

            if plot_train and i_iter % (print_iter*100) == 0:
                fig, ax = plt.subplots(1,1,figsize = (12,4))
                ax.plot(gt_velocity_rolls[0,0,:].cpu().numpy())
                ax.plot(smoothed_velocity_rolls[0,0,:].cpu().detach().numpy())
                ax.set_title("training result at iter "+str(i_iter))
                ax.legend(["ground-truth", "estimated"])

        torch.cuda.empty_cache()

    print("Training finished! Total iter:", training_config["max_train_iter"])