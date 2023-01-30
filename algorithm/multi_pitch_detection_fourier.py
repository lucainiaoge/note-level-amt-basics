import os
import sys
import json
import numpy as np
import cvxpy as cp
import scipy
from scipy.linalg import pinvh
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(directory.parent)
from data.data_utils import resample_audio

from hyperparameters import PITCH_FREQUENCY_GRID, SOURCE_SAMPLE_RATE, PROJECT_SAMPLE_RATE
from utils import normalize, smooth

class CisoidDict:
    def __init__(self, raw_signal_dict, fs, frequency_grid = PITCH_FREQUENCY_GRID, smoothify = False, max_signal_len = 20000, lambda_l1 = 0, pitch_repeat:int = 1, plot = False):
        self.smoothify = smoothify
        self.frequency_grid = frequency_grid
        self.max_signal_len = max_signal_len
        self.pitch_num = len(raw_signal_dict)
        self.pitch_repeat = pitch_repeat
        self.N_frame = len(self.frequency_grid)
        self.frame_analysis_matrix = np.exp(-2*np.pi*1j*frequency_grid.reshape(-1, 1)/fs @ np.arange(self.max_signal_len).reshape(1, -1))
        
        self.cisoid_dict = self.get_cisoid_dict(raw_signal_dict, frequency_grid, plot = plot) #(P, K)
        self.template_A = np.abs(self.cisoid_dict**2).T #(K, P)
        self.output_C = np.eye(self.pitch_num) #(P,P)
        if pitch_repeat > 1:
            self.template_A = np.repeat(self.template_A, pitch_repeat, axis=1) #(K, rP)
            self.output_C = np.repeat(self.output_C, pitch_repeat, axis=1)*1.0 / pitch_repeat #(P, rP)
        
        self.prepare_convex_problem(lambda_l1)
    
    def prepare_convex_problem(self, lambda_l1 = 0):
        self.a_tilde_power = cp.Parameter(self.N_frame, nonneg=True)
        self.b_hat = cp.Variable(self.pitch_num*self.pitch_repeat)
        a_hat = self.template_A @ self.b_hat
        constraints = [self.b_hat >= 0]
        self.difference = (self.a_tilde_power - a_hat)
        
        if lambda_l1 > 0:
            objective = cp.Minimize(cp.sum(cp.square(self.difference)) + lambda_l1 * cp.sum(cp.abs(self.b_hat)))
        else:
            objective = cp.Minimize(cp.sum(cp.square(self.difference)))
        self.prob = cp.Problem(objective, constraints)
    
    def apply_frame_analysis_operator(self, x, spectrum = False):
        N = min(len(x), self.max_signal_len)
        z = self.frame_analysis_matrix[:,:N] @ (x[:N]*(1+0*1j)) / N
        if not spectrum:
            return np.squeeze(np.asarray(z))
        else:
            return np.abs(np.squeeze(np.asarray(z))) ** 2
    
    def apply_frame_analysis_operator_batch(self, x, spectrum = False):
        B, N = x.shape
        N = min(N, self.max_signal_len)
        batch_len = np.count_nonzero(x[:,:N], axis=1)[:, np.newaxis] + 1
        z = (x[:,:N]*(1+0*1j)) @ self.frame_analysis_matrix[:,:N].T / batch_len
        if not spectrum:
            return np.asarray(z)
        else:
            return np.abs(np.asarray(z)) ** 2
    
    def apply_frame_synthesis_operator(self, z, len_signal):
        len_signal = min(len_signal, self.max_signal_len)
        synthesis_matrix = np.matrix(self.frame_analysis_matrix[:,:len_signal]).H
        x = synthesis_matrix @ z
        return np.squeeze(np.asarray(x))
    
    def get_cisoid_dict(self, raw_signal_dict, frequency_grid = PITCH_FREQUENCY_GRID, plot = False):
        N_frame = len(frequency_grid)
        pitch_num = len(raw_signal_dict)
        cisoid_dict = np.zeros((pitch_num, N_frame))*(1+0*1j)
        if plot:
            if pitch_num <= 12:
                fig, ax = plt.subplots(pitch_num,1, sharex=True, sharey=True, figsize=(12,8))
            else:
                fig, ax = plt.subplots(int(pitch_num/8)+1,1, sharex=True, sharey=True, figsize=(12,8))
        for p in range(pitch_num):
            if self.smoothify:
                this_raw_signal = smooth(raw_signal_dict[p], window_len=5)
            else:
                this_raw_signal = raw_signal_dict[p]

            cisoid_dict[p] = self.apply_frame_analysis_operator(this_raw_signal)

            if plot:
                frequency_grid_id = np.arange(N_frame)
                if pitch_num > 12 and p%8==0:
                    plot_id = int(p/8)
                    ax[plot_id].scatter(frequency_grid_id, np.abs(cisoid_dict[p]))
                    ax[plot_id].set_title("amplitudes_"+str(p))
                elif pitch_num <= 12:
                    ax[p].scatter(frequency_grid_id, np.abs(cisoid_dict[p]))
                    ax[p].set_title("amplitudes_"+str(p))

        return cisoid_dict
    
    def single_chord_frame_analysis(self, x, spectrum = False, plot = False):
        if self.smoothify:
            this_raw_signal = smooth(x, window_len=5)
        else:
            this_raw_signal = x
            
        if len(x.shape) == 1:
            a_tilde = self.apply_frame_analysis_operator(this_raw_signal, spectrum = spectrum)
            if plot:
                fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
                ax.plot(abs(a_tilde))
        elif len(x.shape) == 2:
            a_tilde = self.apply_frame_analysis_operator_batch(this_raw_signal, spectrum = spectrum)
            if plot:
                fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
                ax.plot(abs(a_tilde[0]))
            
        return a_tilde

    def single_chord_transcription_powergram(self, x, plot = False, solver = None):
        self.a_tilde_power.value = self.single_chord_frame_analysis(x, spectrum = True, plot = plot)
        if len(x.shape) == 1:
            if solver is None:
                result = self.prob.solve(warm_start=True)
            else:
                result = self.prob.solve(solver=solver, warm_start=True)
            return self.b_hat.value @ self.output_C.T
        
        elif len(x.shape) == 2:
            B, L = x.shape
            result_vec = np.zeros(B, self.pitch_num*self.pitch_repeat)
            for i_batch in range(B):
                if solver is None:
                    result = self.prob.solve(warm_start=True)
                else:
                    result = self.prob.solve(solver=solver, warm_start=True)
                result_vec[i_batch] = self.b_hat.value[:].copy()
            return result_vec @ self.output_C.T
        
    def transcript_with_sparse_NUV(self, x, N_iter = 3, N_sparsity = 10, r2 = 0.001, init_value = 0.5, plot = False):
        input_feature = self.single_chord_frame_analysis(x, spectrum = True, plot = plot)
        if len(x.shape) == 1:
            feat_dim = input_feature.shape[0]
            nuv_q2 = np.zeros(self.pitch_num) + init_value
            for i_iter in range(N_iter):
                precision_mat = pinvh(self.template_A @ np.diag(nuv_q2) @ self.template_A.T + r2 * np.eye(feat_dim)) # (K,K)
                mean_nuv = nuv_q2 * (self.template_A.T @ precision_mat @ input_feature) #(P) · (P,K)x(K,K)x(K) = (P)
                var_nuv = nuv_q2 - nuv_q2**2 * np.einsum("ij,ij->j", self.template_A, precision_mat @ self.template_A)
                nuv_q2 = mean_nuv + var_nuv
            
            max_indices = np.argpartition(nuv_q2,-N_sparsity)[-N_sparsity:]
            result_vec = np.zeros(self.pitch_num)
            result_vec[max_indices] += mean_nuv[max_indices]
            return result_vec
        
    
    def transcript_with_onsets(self, x, onset_ids, offset_ids, onset_offsets = None, N_max = 1600):
        pitch_roll = np.zeros((len(x),self.pitch_num))*(1+0*1j)
        if onset_offsets is None:
            onset_offsets = [0]*len(onset_ids)
        for i in range(len(onset_ids)):
            original_this_id = onset_ids[i]
            this_id = onset_ids[i] + onset_offsets[i]
            next_id = offset_ids[i]
            
            if next_id - this_id > N_max:
                x_chunk = x[this_id:this_id + N_max]
            
            pitch_roll[original_this_id:next_id,:] = self.single_chord_transcription_powergram(x_chunk, plot = False, solver = None)
        
        return pitch_roll

def get_cisoid_dict(template_path, source_fs = SOURCE_SAMPLE_RATE, tgt_fs = PROJECT_SAMPLE_RATE, frequency_grid = PITCH_FREQUENCY_GRID, pitch_repeat:int = 1):
    with open(template_path, 'r') as fin:
        template_data = json.load(fin)
    pitch_num = len(template_data)
    raw_templates = [None for _ in range(pitch_num)]
    for p in range(pitch_num):
        audio = np.array(template_data[p]["audio"])
        raw_templates[p] = resample_audio(audio, source_fs, tgt_fs, axis=0)
        raw_templates[p] = normalize(raw_templates[p], std = 1)
    return CisoidDict(raw_templates, tgt_fs, frequency_grid = frequency_grid, pitch_repeat = pitch_repeat, smoothify = True, plot = False)

def remove_amp_vec_noise(amp_vec, rel_amp_thres = 0.1, rel_amp_thres_octave = 0.4):
    filtered_amp_vec = np.copy(amp_vec)
    max_id = np.argmax(amp_vec)
    if amp_vec[max_id] > 0:
        relative_amp_vec = amp_vec / amp_vec[max_id]
    else:
        relative_amp_vec = amp_vec * 0
        
    if rel_amp_thres > 0:
        filtered_amp_vec[relative_amp_vec < rel_amp_thres] = 0
    
    if rel_amp_thres_octave > 0:
        true_pitch_mask = (relative_amp_vec >= rel_amp_thres_octave)
        true_pitch_indices = np.nonzero(true_pitch_mask)[0]
        octave_indices = np.union1d(true_pitch_indices + 12, true_pitch_indices + 19)
        octave_indices = octave_indices[octave_indices < 88]
        relative_amp_octave = relative_amp_vec[octave_indices]
        false_octave_indices = octave_indices[relative_amp_octave < rel_amp_thres_octave]
        
        filtered_amp_vec[false_octave_indices] = 0
    
    return filtered_amp_vec

def remove_amp_vec_noise_peak_finding(velocity_vec, min_thres = 0.3):
    out_vec = np.zeros_like(velocity_vec)
    max_height = np.max(velocity_vec)
    thres_height = max_height * min_thres
    peak_ids,_ = find_peaks(
        velocity_vec, 
        height = thres_height, 
        threshold = None, 
        distance = None, 
        prominence = None, 
        width = None, 
        wlen = None, rel_height = 0.5, plateau_size = None
    )
    out_vec[peak_ids] = velocity_vec[peak_ids]
    return out_vec
