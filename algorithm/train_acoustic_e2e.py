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

training_config_example = {
    "device": "cuda",
    "checkpoint_dir": "checkpoints",
    "model_name": "AcousticFourier",
    "forward_all_templates": False,
    "learning_rate": 0.0005,
    "start_train_iter": 0,
    "max_train_iter": 10000,
    "print_iter": 100,
    "save_checkpoint_iter": 1000,
    "validate_iter": 20
}

def train_acoustic_e2e(training_config, acoustic_e2e, single_chord_train_loader,
    single_chord_validate_loader = None, save = True, plot_validate = False, plot_train = False
):
    print_iter = training_config["print_iter"]
    device = training_config["device"]
    torch_device = torch.device(device)
    
    if "cuda" in device:
        torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("training using cuda")
        acoustic_e2e.to(torch_device)
    
    optimizer = optim.Adam(acoustic_e2e.parameters(), 
        lr=training_config["learning_rate"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True
    )
    
    i_iter = 0
    if "checkpoint_name" in training_config:
        checkpoint_name = training_config["checkpoint_name"]
        checkpoint_path = os.path.join(training_config["checkpoint_dir"], checkpoint_name)
        if os.path.exists(checkpoint_path):
            i_iter = int(checkpoint_name.split("_")[1])+1
            training_config["start_train_iter"] = i_iter
            acoustic_e2e.load_state_dict(torch.load(checkpoint_path))
            print("loaded checkpoint ", checkpoint_name)
    else:
        training_config["start_train_iter"] = 0
    
    running_loss = 0.0
    running_TP = 0
    running_FP = 0
    running_FN = 0
    for i_iter in range(training_config["start_train_iter"], training_config["max_train_iter"]):
        torch.cuda.empty_cache()
        acoustic_e2e.train()
        optimizer.zero_grad()

        # save checkpoint
        if save and i_iter % training_config["save_checkpoint_iter"] == 0:
            save_checkpoint(acoustic_e2e, training_config["checkpoint_dir"], i_iter, 
                model_name = training_config["model_name"]
            )

        # validate
        if (single_chord_validate_loader is not None) and i_iter % print_iter == 0:
            with torch.no_grad():
                for _ in range(training_config["validate_iter"]):
                    batch_data_dict_val = next(iter(single_chord_validate_loader))
                    acoustic_e2e.eval()
                    audio_input_val = move_data_to_device(batch_data_dict_val["audio"], torch_device)
                    prevchord_input_val = move_data_to_device(batch_data_dict_val["prevchord"], torch_device)
                    velocity_roll_gt_val = move_data_to_device(batch_data_dict_val["velocity_roll"], device = torch_device)
                    velocity_roll_estimate_val = acoustic_e2e(audio_input_val, prevchord_input_val)
                    
                    n_batch_val = velocity_roll_estimate_val.shape[0]
                    for i_batch in range(n_batch_val):
                        est_thres = velocity_roll_estimate_val[i_batch].cpu().numpy().max() * 0.1
                        TP,FP,FN = compare_single_chords(
                            (velocity_roll_gt_val[i_batch].cpu().numpy() > 0)*1, 
                            (velocity_roll_estimate_val[i_batch].cpu().numpy() > est_thres)*1
                        )
                        running_TP += TP
                        running_FP += FP
                        running_FN += FN
                    
                if plot_validate and i_iter % (print_iter*100) == 0:
                    fig, ax = plt.subplots(1,1,figsize = (12,4))
                    ax.plot(velocity_roll_gt_val[0].cpu().numpy())
                    ax.plot(velocity_roll_estimate_val[0].cpu().numpy())
                    ax.set_title("validate result (velocity) at iter "+str(i_iter))
                    ax.legend(["ground-truth", "estimated"])
                    
        # forward data and calculate loss and backward
        batch_data_dict = next(iter(single_chord_train_loader))
        audio_input = move_data_to_device(batch_data_dict["audio"], torch_device)
        prevchord_input = move_data_to_device(batch_data_dict["prevchord"], torch_device)
        velocity_roll_gt = move_data_to_device(batch_data_dict["velocity_roll"], device = torch_device)
        velocity_roll_estimate = acoustic_e2e(audio_input, prevchord_input)
        
        mask_all = (velocity_roll_gt * 0) + 1
        # mask_tgt = (velocity_roll_gt > 0) #torch.logical_or((velocity_roll_gt > 0), (torch.randn_like(velocity_roll_gt) > 1))
        loss = 2*bce(velocity_roll_estimate, velocity_roll_gt, mask_all)
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(acoustic_e2e.parameters(), 0.01)
            running_loss += loss.item()
            optimizer.step()
        optimizer.zero_grad()
                    
        # print statistics
        if i_iter % print_iter == 0 and i_iter > 0:
            if running_TP > 0:
                eval_P = 1.0*running_TP / (running_TP + running_FP)
                eval_R = 1.0*running_TP / (running_TP + running_FN)
                eval_F1 = 2*eval_P*eval_R / (eval_P + eval_R)
            else:
                eval_P = 0
                eval_R = 0
                eval_F1 = 0
            
            if single_chord_validate_loader is not None:
                print(f'[{i_iter:5d}] loss: {running_loss/print_iter:.5f}; validate P,R,F1: {eval_P:.4f},{eval_R:.4f},{eval_F1:.4f}')
            else:
                print(f'[{i_iter:5d}] loss: {running_loss/print_iter:.5f}')
            running_loss = 0
            running_TP = 0
            running_FP = 0
            running_FN = 0

            if plot_train and i_iter % (print_iter*100) == 0:
                fig, ax = plt.subplots(1,1,figsize = (12,4))
                ax.plot(velocity_roll_gt[0].cpu().numpy())
                ax.plot(velocity_roll_estimate[0].cpu().detach().numpy())
                ax.set_title("training result at iter "+str(i_iter))
                ax.legend(["ground-truth", "estimated"])

        torch.cuda.empty_cache()

    print("Training finished! Total iter:", training_config["max_train_iter"])

