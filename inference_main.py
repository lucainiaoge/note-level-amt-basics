import argparse

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from data.data_utils import read_audio, rectify_piano_roll, rectify_chord_onset_offset_seq, maestro_piano_roll_to_midi, amp2velocity_appox
from data.data_utils import maestro_read_midi_to_note_list, maestro_read_midi_to_piano_roll, get_concurrent_notes_from_piano_roll
from data.data_utils import pitch_velocity_rolls_onsets_offsets_to_piano_roll
from data.data_utils import clip_or_pad_to_fixed_len
from data.data_utils import get_pitch_activation_roll_mode

from model.unrolled_mse_net import UnrolledMSENet
from model.transduction_net import TransductionFC

from algorithm.onset_offset_detection import get_onset_offsets
from algorithm.multi_pitch_detection_fourier import get_cisoid_dict, remove_amp_vec_noise
from algorithm.transduction_model import single_chord_hmm_viterbi_decoding, str_decisions_to_pitch_rolls

from utils import normalize, smooth, move_data_to_device
from hyperparameters import SOURCE_SAMPLE_RATE, PROJECT_SAMPLE_RATE, PITCH_FREQUENCY_GRID, LOWEST_PITCH, FIXED_LEN
from hyperparameters import MAESTRO_TICK_PER_SEC
from hyperparameters import ACOUSTIC_NN_CKPT_PATH, TRNASDUCTION_NN_CKPT_PATH, TRNASDUCTION_NN_FOR_E2E_ACOUSTIC_MODEL_CKPT_PATH
from hyperparameters import TEMPLATE_PATH, NOTE_TRANSITION_DICT_PATH, CISOID_A_TEMPLATE_CKPT_PATH
from hyperparameters import MODEL_TYPE_BASELINE, MODEL_TYPE_UNROLLEDNET

VEL_ROLL_SAVE_DIR_BASELINE = os.path.join("results","vel_rolls_after_acoustic_fourier")
VEL_ROLL_SAVE_DIR_UNROLLEDNET = os.path.join("results","vel_rolls_after_acoustic_unrollednet")
GT_MIDI_DIR = os.path.join("data","midi_no_cc","2004")

LOAD_TRAINED_TEMPLATE_DICT = False
# choose model_type from ["baseline", "unrollednet"]
def prepare_model_elements(model_type, verbose = False):
    # load pitch templates for acoustic model
    if model_type == MODEL_TYPE_BASELINE:
        acoustic_model = get_cisoid_dict(TEMPLATE_PATH)
        if os.path.exists(CISOID_A_TEMPLATE_CKPT_PATH) and LOAD_TRAINED_TEMPLATE_DICT:
            acoustic_model.template_A = np.load(CISOID_A_TEMPLATE_CKPT_PATH)
            acoustic_model.prepare_convex_problem()
        if verbose:
            print("Preparation: loading template audio and creating cisoid dictionary...")
    elif model_type == MODEL_TYPE_UNROLLEDNET:
        acoustic_model = UnrolledMSENet()
        acoustic_model.load_state_dict(torch.load(ACOUSTIC_NN_CKPT_PATH))
        acoustic_model.eval()
        if verbose:
            print("Preparation: loading acoustic model checkpoint...")
    else:
        assert 0, "choose model_type from ['baseline', 'unrollednet']"
        
    # load transmit probability estimation NN in the transduction model
    if verbose:
        print("Preparation: loading transmit probability estimation NN in the transduction model...")
        
    transduction_nn = TransductionFC()
    if model_type == MODEL_TYPE_BASELINE:
        transduction_nn.load_state_dict(torch.load(TRNASDUCTION_NN_CKPT_PATH))
    elif model_type == MODEL_TYPE_UNROLLEDNET:
        transduction_nn.load_state_dict(torch.load(TRNASDUCTION_NN_FOR_E2E_ACOUSTIC_MODEL_CKPT_PATH))
    else:
        assert 0, "choose model_type from ['baseline', 'unrollednet']"
    transduction_nn.eval()

    # load note transition dict in the transduction model
    if verbose:
        print("Preparation: loading note transition dict in the transduction model...")
    with open(NOTE_TRANSITION_DICT_PATH, 'r') as fin:
        note_transition_dict = json.load(fin)
    
    return acoustic_model, transduction_nn, note_transition_dict

def piano_roll_smooth_with_gt_notes(piano_roll, gt_note_list, lowest_pitch = 21):
    new_piano_roll = piano_roll[:].copy()
    for note in gt_note_list:
        onset = note.onset
        offset = min(note.offset, len(piano_roll))
        pitch_index = note.pitch - lowest_pitch
        new_piano_roll[onset:offset,pitch_index] = np.mean(piano_roll[onset:offset,pitch_index])
    return new_piano_roll

def transcribe_audio(audio_path, model_type, acoustic_model, transduction_nn, note_transition_dict, 
    amp2vel_mapping = None, vel_roll_path = None,
    segment_with_gt = False, gt_acoustic = False, 
    no_transition = False, no_hmm = False, gt_smoothing = False, 
    verbose = False
):
    # setups: read audio
    if verbose:
        print("-= Step 0: Reading audio =-")
    waveform, fs = read_audio(audio_path, PROJECT_SAMPLE_RATE)
    waveform = normalize(waveform, std = 1)

    # step 1: onset-offset segmentation
    if verbose:
        print("-= Step 1: Running onset-offset segmentation =-")
    
    if segment_with_gt or gt_acoustic:
        piece_name = os.path.split(audio_path)[-1][:-4]
        midi_path = os.path.join(GT_MIDI_DIR, piece_name+".midi")
        gt_piano_roll, tick_per_sec = maestro_read_midi_to_piano_roll(midi_path, pitch_range = 88, lowest_pitch = 21, min_len = 20)
        _,onset_ids_midi, offset_ids_midi = get_concurrent_notes_from_piano_roll(gt_piano_roll, lowest_pitch = 21, min_len = 2)
        
    if not segment_with_gt:
        onset_ids_audio, offset_ids_audio = get_onset_offsets(waveform, fs)
        assert len(onset_ids_audio) == len(offset_ids_audio), "abnormal onset-offset detection: num onsets should be same as offsets"
        n_segments = len(onset_ids_audio)
        onsets_sec = 1.0*onset_ids_audio/PROJECT_SAMPLE_RATE
        offsets_sec = 1.0*offset_ids_audio/PROJECT_SAMPLE_RATE
        
        tick_per_sec = MAESTRO_TICK_PER_SEC
        onset_ids_midi = (onsets_sec * tick_per_sec).astype(int)
        offset_ids_midi = (offsets_sec * tick_per_sec).astype(int)
    else:
        n_segments = len(onset_ids_midi)
        onsets_sec = 1.0*np.array(onset_ids_midi)/tick_per_sec
        offsets_sec = 1.0*np.array(offset_ids_midi)/tick_per_sec
        onset_ids_audio = (onsets_sec*PROJECT_SAMPLE_RATE).astype(int)
        offset_ids_audio = (offsets_sec*PROJECT_SAMPLE_RATE).astype(int)
    if verbose:
        print("num segments =", n_segments)

    # step 2: for each segment, run acoustic model
    if vel_roll_path is None:
        if verbose:
            print("-= Step 2: Running acoustic model =-")
        est_velocity_rolls = [None]*n_segments
        if not gt_acoustic:
            for i_seg in range(n_segments):
                waveform_seg = waveform[onset_ids_audio[i_seg]:offset_ids_audio[i_seg]]
                if model_type == MODEL_TYPE_BASELINE:
                    est_velocity_rolls[i_seg] = np.abs(acoustic_model.single_chord_transcription_powergram(waveform_seg, plot = False))
                    # est_velocity_rolls[i_seg] = np.abs(acoustic_model.transcript_with_sparse_NUV(waveform_seg, N_iter = 3, N_sparsity = 10, r2 = 0.005, init_value = 0.5, plot = False))
                elif model_type == MODEL_TYPE_UNROLLEDNET:
                    waveform_seg = clip_or_pad_to_fixed_len(waveform_seg, FIXED_LEN)
                    waveform_seg = move_data_to_device(waveform_seg, "cpu")
                    with torch.no_grad():
                        est_velocity_rolls[i_seg] = acoustic_model(waveform_seg, None).numpy()
                else:
                    assert 0, "choose model_type from ['baseline', 'unrollednet']"
        else:
            max_tick, _ = gt_piano_roll.shape
            for i_seg in range(n_segments):
                this_onset_ids_midi = min(int(onsets_sec[i_seg] * tick_per_sec), max_tick)
                this_offset_ids_midi = min(int(offsets_sec[i_seg] * tick_per_sec), max_tick)
                piano_roll_seg = (gt_piano_roll[this_onset_ids_midi:this_offset_ids_midi] > 0) * 1
                est_velocity_rolls[i_seg] = get_pitch_activation_roll_mode(piano_roll_seg)
                
        est_velocity_rolls = np.stack(est_velocity_rolls, axis=0) #(n_segments, n_pitch)
        piece_name = audio_path.split(os.sep)[-1][:-4]
        vel_roll_save_path = os.path.join(VEL_ROLL_SAVE_DIR_BASELINE, piece_name+".npy")
        with open(vel_roll_save_path, 'wb') as f:
            np.save(f, est_velocity_rolls)
    else:
        if verbose:
            print("-= Step 2: Reading transcribed velocity rolls =-")
        est_velocity_rolls = np.load(vel_roll_path)

    # step 3: run transduction model
    if not no_hmm:
        if verbose:
            print("-= Step 3.1: Running transmit probability estimation =-")
        torch_est_velocity_rolls =  move_data_to_device(est_velocity_rolls, "cpu")
        torch_durs = torch.unsqueeze(move_data_to_device(offsets_sec - onsets_sec, "cpu"), 1)
        torch_noisy_velocity_roll_input = torch.cat([torch_est_velocity_rolls, torch_durs], dim = 1)
        with torch.no_grad():
            est_pitch_prob_rolls = transduction_nn(torch_noisy_velocity_roll_input).numpy()
            est_pitch_prob_rolls[est_pitch_prob_rolls < 0.1] = 0

        if verbose:
            print("-= Step 3.2: Running HMM transduction model =-")
        if not no_transition:
            decisions = single_chord_hmm_viterbi_decoding(est_pitch_prob_rolls, note_transition_dict, verbose = False)
            opt_pitch_rolls = str_decisions_to_pitch_rolls(decisions, num_pitches = 88)
        else:
            est_pitch_prob_rolls[est_pitch_prob_rolls < 0.5] = 0
            opt_pitch_rolls = (est_pitch_prob_rolls > 0)
    else:
        if verbose:
            print("-= Step 3: No transduction model, thresholding the results =-")
        opt_pitch_rolls = (est_velocity_rolls > 0)
        for i_roll in range(est_velocity_rolls.shape[0]):
            opt_pitch_rolls[i_roll] = (remove_amp_vec_noise(est_velocity_rolls[i_roll], rel_amp_thres = 0.1, rel_amp_thres_octave = 0.4)>0)
    
    
    opt_pitch_rolls = rectify_chord_onset_offset_seq(
        opt_pitch_rolls, onset_ids_midi, offset_ids_midi, 
    )
    est_piano_roll = pitch_velocity_rolls_onsets_offsets_to_piano_roll(
        opt_pitch_rolls, est_velocity_rolls, onsets_sec, offsets_sec
    )
    if gt_smoothing:
        piece_name = os.path.split(audio_path)[-1][:-4]
        midi_path = os.path.join(GT_MIDI_DIR, piece_name+".midi")
        gt_note_list, tick_per_sec = maestro_read_midi_to_note_list(midi_path, min_len = 20)
        est_piano_roll = piano_roll_smooth_with_gt_notes(est_piano_roll, gt_note_list, lowest_pitch = 21)
    
    if amp2vel_mapping is not None:
        est_piano_roll = amp2vel_mapping(est_piano_roll).astype(int)
    return rectify_piano_roll(est_piano_roll, reduce_map = "max")

# example running: 
# python inference_main.py data/synth/2004/MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_05_Track05_wav.wav test.midi baseline -v
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Give audio path to transcribe and the saving path.")
    parser.add_argument("audio_path", type=str, help="the path of the audio (relative to the code dir)")
    parser.add_argument("save_path", type=str, help="the path to save the midi as transcription result")
    parser.add_argument("model_type", type=str, help="choose model_type from ['baseline', 'unrollednet']")
    parser.add_argument("-v", "--verbose", action="store_true", help="activate verbose mode with '-v'")
    args = parser.parse_args()
    
    # do preparations
    acoustic_model, transduction_nn, note_transition_dict = prepare_model_elements(args.model_type, args.verbose)
    
    # do transcription
    if args.verbose:
        print("Transcription starts...")
    est_piano_roll = transcribe_audio(
        args.audio_path, args.model_type, acoustic_model, transduction_nn, note_transition_dict, amp2vel_mapping = amp2velocity_appox, 
        verbose = args.verbose
    )

    # save the piano roll to midi
    maestro_piano_roll_to_midi(est_piano_roll, args.save_path, 
        lowest_pitch = LOWEST_PITCH, amp2vel_mapping = None
    )

