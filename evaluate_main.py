import argparse
import os
import json
import numpy as np
import mido
import matplotlib.pyplot as plt

import numpy as np
import mir_eval
import matplotlib.pyplot as plt

from data.data_utils import remove_audio_filename_extension, remove_midi_filename_extension, maestro_piano_roll_to_midi
from data.data_utils import maestro_piano_roll_to_note_list, maestro_onset_offset_pitch_list_to_piano_roll
from data.data_utils import maestro_get_tick_per_second, maestro_midi_to_note_list
from data.data_utils import decompose_note_list, maestro_read_midi_to_piano_roll, amp2velocity_appox

from inference_main import transcribe_audio, prepare_model_elements
from utils import midi_to_hz
from hyperparameters import SUPPORTED_MIDI_TYPES, SUPPORTED_AUDIO_TYPES
from hyperparameters import MAESTRO_TICK_PER_SEC, LOWEST_PITCH
from hyperparameters import TEMPLATE_PATH, TRNASDUCTION_NN_CKPT_PATH, NOTE_TRANSITION_DICT_PATH
from hyperparameters import MODEL_TYPE_BASELINE, MODEL_TYPE_UNROLLEDNET

def clipped_translation(vec, k):
    len_vec = len(vec)
    out_vec = np.zeros(len_vec)
    if k >= 0:
        out_vec[k:] = vec[:len_vec-k]
    else:
        out_vec[:len_vec+k] = vec[-k:]
    return out_vec


ROUND1_SEARCH_WINDOW = 2 * MAESTRO_TICK_PER_SEC
ROUND1_SEARCH_STEP = 8
ROUND2_SEARCH_WINDOW = int(0.5 * MAESTRO_TICK_PER_SEC)
ROUND2_SEARCH_STEP = 2
ROUND3_SEARCH_WINDOW = int(0.05 * MAESTRO_TICK_PER_SEC)
ROUND3_SEARCH_STEP = 1
def calculate_transcription_note_metric_note_list(gt_note_list, est_note_list, 
    tick_per_sec = MAESTRO_TICK_PER_SEC, lowest_pitch = 21, auto_time_align = False,
    onset_tolerance = 0.05, pitch_tolerance=50.0, offset_ratio=0.2, offset_min_tolerance=0.05, strict=False, beta=1.0,
    plot = False
):
    gt_pitch_list, _, gt_onset_list, gt_offset_list = decompose_note_list(gt_note_list)
    est_pitch_list, _, est_onset_list, est_offset_list = decompose_note_list(est_note_list)
    
    gt_intervals = np.array([gt_onset_list, gt_offset_list]).T
    est_intervals = np.array([est_onset_list, est_offset_list]).T
    
    # this is an algorithm to align the est and gt, when there is a shift of onset/offset in est
    if auto_time_align:
        if np.min(est_intervals) < 2 and (np.min(gt_intervals) - np.min(est_intervals)) > 0.05 * tick_per_sec:
            est_intervals = est_intervals + np.min(gt_intervals)
        
        max_time = max(np.max(gt_intervals), np.max(est_intervals))
        gt_time_roll = np.zeros(max_time)
        est_time_roll = np.zeros(max_time)
        gt_time_roll[gt_intervals[:,0]] = 1
        est_time_roll[est_intervals[:,0]] = 1
        gt_time_roll = np.convolve(gt_time_roll, np.hanning(9), mode="same")
        est_time_roll = np.convolve(est_time_roll, np.hanning(9), mode="same")
        L = len(gt_time_roll)

        if len(gt_time_roll) < 2*ROUND1_SEARCH_WINDOW: # this method has very high runtime when length of time_roll is high
            correlate_roll = np.correlate(est_time_roll, gt_time_roll, mode="same") # pad_on_est = [L/2,...,0,...,-L/2], id = [0,...,L/2,...,L]
            max_corr_id = np.argmax(correlate_roll)
            pad = int(L/2 - max_corr_id)
        else: # this method has 3 round search for best match
            correlate_roll = np.zeros(L) # pad_on_est = [L/2,...,0,...,-L/2], id = [0,...,L/2,...,L]
            # round 1:
            for k in range(-ROUND1_SEARCH_WINDOW, ROUND1_SEARCH_WINDOW, ROUND1_SEARCH_STEP): # zeros to pad
                corr_roll_k = int(L/2 - k)
                correlate_roll[corr_roll_k] = np.inner(clipped_translation(est_time_roll, k), gt_time_roll)
            round1_max = np.argmax(correlate_roll)
            round1_pad = int(L/2 - round1_max)
            # round 2:
            for k in range(round1_pad-ROUND2_SEARCH_WINDOW, round1_pad+ROUND2_SEARCH_WINDOW, ROUND2_SEARCH_STEP): # zeros to pad
                corr_roll_k = int(L/2 - k)
                if correlate_roll[corr_roll_k] == 0:
                    correlate_roll[corr_roll_k] = np.inner(clipped_translation(est_time_roll, k), gt_time_roll)
            round2_max = np.argmax(correlate_roll)
            round2_pad = int(L/2 - round2_max)
            # round 3:
            for k in range(round2_pad-ROUND3_SEARCH_WINDOW, round2_pad+ROUND3_SEARCH_WINDOW, ROUND3_SEARCH_STEP): # zeros to pad
                corr_roll_k = int(L/2 - k)
                if correlate_roll[corr_roll_k] == 0:
                    correlate_roll[corr_roll_k] = np.inner(clipped_translation(est_time_roll, k), gt_time_roll)
            round3_max = np.argmax(correlate_roll)
            pad = int(L/2 - round3_max)

        est_intervals = est_intervals[:] + pad
    
    gt_intervals = gt_intervals / tick_per_sec
    est_intervals = est_intervals / tick_per_sec

    gt_pitches = midi_to_hz(np.array(gt_pitch_list))
    est_pitches = midi_to_hz(np.array(est_pitch_list))
    
    if plot:
        gt_roll_bool = maestro_onset_offset_pitch_list_to_piano_roll(
            gt_onset_list, gt_offset_list, gt_pitch_list, 88, lowest_pitch
        )
        est_roll_bool = maestro_onset_offset_pitch_list_to_piano_roll(
            est_onset_list, est_offset_list, est_pitch_list, 88, lowest_pitch
        )
        fig, ax = plt.subplots(2,1, figsize=(12,6))
        show_len = min(len(gt_roll_bool), len(est_roll_bool))
        show_len = min(24000, show_len)
        ax[0].matshow(gt_roll_bool[:show_len:48].T, origin = "lower")
        ax[0].set_title("ground-truth pitch piano roll")
        ax[1].matshow(est_roll_bool[:show_len:48].T, origin = "lower")
        ax[1].set_title("estimated pitch piano roll")
    
    (precision, recall, F1, _) = mir_eval.transcription.precision_recall_f1_overlap(
        gt_intervals, gt_pitches, est_intervals, est_pitches, 
        onset_tolerance, pitch_tolerance, offset_ratio, offset_min_tolerance, strict, beta
    )
    return precision, recall, F1

def calculate_transcription_note_metric_piano_roll(gt_piano_roll, est_piano_roll, 
    tick_per_sec = MAESTRO_TICK_PER_SEC, lowest_pitch = 21, auto_time_align = False,
    onset_tolerance = 0.05, pitch_tolerance=50.0, offset_ratio=0.2, offset_min_tolerance=0.05, strict=False, beta=1.0,
    plot = False
):
    gt_note_list = maestro_piano_roll_to_note_list(gt_piano_roll, lowest_pitch)
    est_note_list = maestro_piano_roll_to_note_list(est_piano_roll, lowest_pitch)

    return calculate_transcription_note_metric_note_list(gt_note_list, est_note_list, 
        tick_per_sec = tick_per_sec, lowest_pitch = lowest_pitch, auto_time_align = auto_time_align,
        onset_tolerance = onset_tolerance, pitch_tolerance=pitch_tolerance, 
        offset_ratio=offset_ratio, offset_min_tolerance=offset_min_tolerance, 
        strict=strict, beta=beta, plot = plot
    )

def calculate_transcription_note_metric_midi(gt_midi_path, est_midi_path, 
    tick_per_sec = MAESTRO_TICK_PER_SEC, lowest_pitch = 21, auto_time_align = False,
    onset_tolerance = 0.05, pitch_tolerance=50.0, offset_ratio=0.2, offset_min_tolerance=0.05, strict=False, beta=1.0,
    plot = False
):
    gt_midi_obj = mido.MidiFile(gt_midi_path)
    est_midi_obj = mido.MidiFile(est_midi_path)

    gt_tick_per_sec = maestro_get_tick_per_second(gt_midi_obj)
    est_tick_per_sec = maestro_get_tick_per_second(est_midi_obj)
    assert gt_tick_per_sec == est_tick_per_sec, "tick per sec of the two midi should be the same"

    gt_note_list = maestro_midi_to_note_list(gt_midi_obj)
    est_note_list = maestro_midi_to_note_list(est_midi_obj)

    return calculate_transcription_note_metric_note_list(gt_note_list, est_note_list, 
        tick_per_sec = tick_per_sec, lowest_pitch = lowest_pitch, auto_time_align = auto_time_align,
        onset_tolerance = onset_tolerance, pitch_tolerance=pitch_tolerance, 
        offset_ratio=offset_ratio, offset_min_tolerance=offset_min_tolerance, 
        strict=strict, beta=beta, plot = plot
    )

def compare_single_chords(gt_pitch_roll, est_pitch_roll):
    num_tp = np.sum(np.logical_and(gt_pitch_roll!=0, est_pitch_roll!=0).astype(int))
    num_fp = np.sum(np.logical_and(gt_pitch_roll==0, est_pitch_roll!=0).astype(int))
    num_fn = np.sum(np.logical_and(gt_pitch_roll!=0, est_pitch_roll==0).astype(int))
    return num_tp, num_fp, num_fn


def calculate_P_R_F1_given_counts(TP, FP, FN):
    if TP != 0:
        P = 1.0*TP / (TP + FP)
        R = 1.0*TP / (TP + FN)
        F1 = 2*P*R / (P+R)
    else:
        P = 0
        R = 0
        F1 = 0
    return P,R,F1

def calculate_single_chord_metric(gt_pitch_rolls, est_pitch_rolls, calculate_var_n_chord = 0):
    assert len(gt_pitch_rolls) == len(est_pitch_rolls), "input pitch roll lists should have the same length"
    N_chords = len(gt_pitch_rolls)
    num_tp_all = 0
    num_fp_all = 0
    num_fn_all = 0
    for i_roll in range(N_chords):
        num_tp, num_fp, num_fn = compare_single_chords(gt_pitch_rolls[i_roll], est_pitch_rolls[i_roll])
        num_tp_all += num_tp
        num_fp_all += num_fp
        num_fn_all += num_fn
    
    P,R,F1 = calculate_P_R_F1_given_counts(num_tp_all, num_fp_all, num_fn_all)
    
    if calculate_var_n_chord > 0:
        num_tp_temp = 0
        num_fp_temp = 0
        num_fn_temp = 0
        P_batch = []
        R_batch = []
        F1_batch = []
        for i_roll in range(N_chords):
            num_tp, num_fp, num_fn = compare_single_chords(gt_pitch_rolls[i_roll], est_pitch_rolls[i_roll])
            num_tp_temp += num_tp
            num_fp_temp += num_fp
            num_fn_temp += num_fn
            if i_roll % calculate_var_n_chord == 0:
                P_temp,R_temp,F1_temp = calculate_P_R_F1_given_counts(num_tp_temp, num_fp_temp, num_fn_temp)
                P_batch.append(P_temp)
                R_batch.append(R_temp)
                F1_batch.append(F1_temp)
                num_tp_temp = 0
                num_fp_temp = 0
                num_fn_temp = 0
                
        P_batch = np.array(P_batch)
        R_batch = np.array(R_batch)
        F1_batch = np.array(F1_batch)
        return P,R,F1,P_batch.std(),R_batch.std(),F1_batch.std()
    else:
        return P,R,F1

def batch_AMT_with_eval(audio_dir, save_dir, model_type, gt_midi_dir = None, eval_piece_names = None, 
    segment_with_gt = False, gt_acoustic = False, no_transition = False, no_hmm = False, gt_smoothing = False, 
    onset_tolerance = 0.05, pitch_tolerance=50.0, offset_ratio=0.2, offset_min_tolerance=0.05,
    verbose = False
):
    # do preparations
    cisoid_dict, transduction_nn, note_transition_dict = prepare_model_elements(model_type, verbose)
    
    # parse path information and detect running mode
    audio_names = os.listdir(audio_dir)
    piece_names_by_audio = [remove_audio_filename_extension(filename) for filename in audio_names]
    
    if gt_midi_dir is not None and os.path.exists(gt_midi_dir):
        evaluate_flag = True
        midi_names = os.listdir(gt_midi_dir)
        piece_names_by_midi = [remove_midi_filename_extension(filename) for filename in midi_names]
        if verbose:
            print("Groud-truth midi directory detected. Evaluate model switched on.")
    else:
        evaluate_flag = False
        midi_names = []
        piece_names_by_midi = []
        if verbose:
            print("Groud-truth midi directory not detected. Evaluate model switched off.")
    
    valid_piece_names = []
    valid_audio_names = []
    valid_midi_names = []
    for i_file,piece_name in enumerate(piece_names_by_audio):
        if evaluate_flag:
            run_this_piece_flag = (piece_name is not None) and (piece_name in piece_names_by_midi)
            if type(eval_piece_names) == list:
                if not piece_name in eval_piece_names:
                    run_this_piece_flag = False
        else:
            run_this_piece_flag = (piece_name is not None)
        
        if run_this_piece_flag:
            valid_piece_names.append(piece_name)
            for possible_ext in SUPPORTED_AUDIO_TYPES:
                if piece_name + possible_ext in audio_names:
                    valid_audio_names.append(piece_name + possible_ext)
                    break
            if evaluate_flag:
                for possible_ext in SUPPORTED_MIDI_TYPES:
                    if piece_name + possible_ext in midi_names:
                        valid_midi_names.append(piece_name + possible_ext)
                        break
    
    # do transcription and evaluation
    eval_results = {}
    eval_save_path = os.path.join(save_dir, "mir_eval_notes.json")
    for i_file_valid, piece_name in enumerate(valid_piece_names):
        # get the file paths
        audio_name = valid_audio_names[i_file_valid]
        if evaluate_flag:
            midi_name = valid_midi_names[i_file_valid]
            midi_path = os.path.join(gt_midi_dir, midi_name)
            if not os.path.exists(midi_path):
                midi_path = None
        else:
            midi_path = None
        audio_path = os.path.join(audio_dir, audio_name)
        save_path = os.path.join(save_dir, piece_name+".midi")
        
        # run the transcription model
        print("Transcription "+str(i_file_valid)+": "+piece_name+"...")
        est_piano_roll = transcribe_audio(
            audio_path, model_type, cisoid_dict, transduction_nn, note_transition_dict, amp2vel_mapping = amp2velocity_appox,
            vel_roll_path = None, segment_with_gt = segment_with_gt, gt_acoustic = gt_acoustic, 
            no_transition = no_transition, no_hmm = no_hmm, gt_smoothing = gt_smoothing, verbose = False
        )
        
        # save the piano roll to midi
        maestro_piano_roll_to_midi(est_piano_roll, save_path, 
            lowest_pitch = LOWEST_PITCH, amp2vel_mapping = None
        )
        
        # run evaluation
        if midi_path is not None:
            gt_piano_roll, tick_per_sec = maestro_read_midi_to_piano_roll(midi_path)
            P,R,F1 = calculate_transcription_note_metric_piano_roll(gt_piano_roll, est_piano_roll, auto_time_align = False,
                onset_tolerance = onset_tolerance, pitch_tolerance = pitch_tolerance, 
                offset_ratio = offset_ratio, offset_min_tolerance = offset_min_tolerance, 
                strict = False, beta = 1.0, plot = False
            )
            eval_results[piece_name] = (P,R,F1)
            print("Transcription eval result (P,R,F1):", (P,R,F1))
            with open(eval_save_path, "w") as outfile:
                json.dump(eval_results, outfile)
    
    N_eval = len(eval_results)
    PRF1_results = np.array(list(eval_results.values()))
    ave_results = PRF1_results.mean(axis=0)
    ave_std = PRF1_results.std(axis=0)
    print(f"Average P,R,F1: {ave_results[0]:.5f}±{ave_std[0]:.5f}, {ave_results[1]:.5f}±{ave_std[1]:.5f}, {ave_results[2]:.5f}±{ave_std[2]:.5f}")
    print("done!")

# example running: 
# python evaluate_main.py data/synth/2004 results/baseline_2004 --gt_midi_dir data/midi_no_cc/2004 -v
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Give audio directory to transcribe and the saving directory; specify the ground-truth directory for mid_eval evaluation."
    )
    parser.add_argument("audio_dir", type=str, 
        help="the directory of the audio files (relative to the code dir)"
    )
    parser.add_argument("save_dir", type=str, 
        help="the directory to save the midis as transcription results"
    )
    parser.add_argument("model_type", type=str, 
        help="choose model_type from ['baseline', 'unrollednet']"
    )
    parser.add_argument("--gt_midi_dir", type=str, required=True,
        help="the directory of the ground-truth midis (should have same names as audios before .mid or .midi)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="activate verbose mode with '-v'")
    args = parser.parse_args()
    
    batch_AMT_with_eval(args.audio_dir, args.save_dir, args.model_type, gt_midi_dir = args.gt_midi_dir, verbose = args.verbose)
    

