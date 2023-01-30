import os
import json
import numpy as np
import mido
from itertools import chain, combinations

import matplotlib.pyplot as plt

from data.data_utils import remove_midi_filename_extension, maestro_get_tick_per_second, remove_short_notes, pitch_vec_to_piano_roll
from data.data_utils import maestro_midi_to_note_list, maestro_note_list_to_piano_roll,  get_concurrent_notes_from_piano_roll
from data.data_utils import maestro_piano_roll_to_note_list
from hyperparameters import SOURCE_SAMPLE_RATE, PROJECT_SAMPLE_RATE, PITCH_FREQUENCY_GRID, PITCH_NUM, MAESTRO_TICK_PER_SEC

def chord_tuple_list_to_string(tup, baseline = 0):
    return str([p-baseline for p in tup])[1:-1].replace(" ", "")

def chord_string_to_list(chord_string):
    if len(chord_string) > 0:
        notes = chord_string.split(",")
        notes = [int(note) for note in notes]
    else:
        notes = []
    return notes

''' format: {pitches: {next_pitches: number_occurred}}
example:
{
    "41":{
        "41": 2,
        "41,44": 7,
        ...
    },
    ...
}
'''
def get_note_transition_dict(midi_dir, pitch_range = 88, lowest_pitch = 21, min_len = 20):
    midi_names = os.listdir(midi_dir)
    note_transition_dict = {}
    for midi_name in midi_names:
        piece_name = remove_midi_filename_extension(midi_name)
        if piece_name is not None:
            midi_path = os.path.join(midi_dir, midi_name)

            midi_obj = mido.MidiFile(midi_path)
            tick_per_sec = maestro_get_tick_per_second(midi_obj)

            note_list = maestro_midi_to_note_list(midi_obj)
            note_list = remove_short_notes(note_list, min_len = min_len)
            piano_roll = maestro_note_list_to_piano_roll(note_list, pitch_range = pitch_range, lowest_pitch = lowest_pitch)
            concurrent_notes_list, concurrent_notes_onsets, concurrent_notes_offsets = get_concurrent_notes_from_piano_roll(
                piano_roll, lowest_pitch = lowest_pitch
            )

            for i_seg in range(len(concurrent_notes_list)-1):
                this_pitches = concurrent_notes_list[i_seg].pitches #pitch_vec_to_piano_roll(concurrent_notes_list[i_seg].pitches, pitch_range = 88, lowest_pitch = 21)
                next_pitches = concurrent_notes_list[i_seg+1].pitches #pitch_vec_to_piano_roll(concurrent_notes_list[i_seg+1].pitches, pitch_range = 88, lowest_pitch = 21)
                this_pitches = chord_tuple_list_to_string(this_pitches, baseline = lowest_pitch)
                next_pitches = chord_tuple_list_to_string(next_pitches, baseline = lowest_pitch)
                if this_pitches not in note_transition_dict:
                    note_transition_dict[this_pitches] = {next_pitches:1}
                else:
                    if next_pitches not in note_transition_dict[this_pitches]:
                        note_transition_dict[this_pitches][next_pitches] = 1
                    else:
                        note_transition_dict[this_pitches][next_pitches] += 1
            print(piece_name, "done")
    return note_transition_dict

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def get_num_concurrent_octaves(pitches):
    n_octave = 0
    anchor = pitches[0]
    for p in pitches:
        if p - anchor >= 12:
            anchor = p
            n_octave += 1
    return n_octave

def pitches_sanity_check(pitches):
    if len(pitches) > 10:
        return False
    elif get_num_concurrent_octaves(pitches) >= 4:
        return False
    else:
        return True

def get_roll_weight(binary_probs, prob_thres = 0.0001):
    return np.sum(np.log2(binary_probs+prob_thres))
    
def ranked_list_diff(list_large, list_subtract):
    return [x for x in list_large if x not in list_subtract]

def get_support_set(pitch_prob_roll, max_num_pitch = 12):
    support_set = np.nonzero(pitch_prob_roll)[0]
    if len(support_set) > max_num_pitch:
        nonzero_probs = pitch_prob_roll[support_set]
        sorted_ids = np.argsort(nonzero_probs)[::-1][:max_num_pitch]
        support_set = support_set[sorted_ids]
    return support_set

def list_argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def remove_less_possible_choices(choice_strs, logprobs, max_num_choice = 64):
    assert len(logprobs) == len(choice_strs), "length of logprobs should be the same as choice strings"
    if len(logprobs) > max_num_choice:
        sorted_ids = list_argsort(logprobs)[-max_num_choice:]
        choice_strs = [choice_strs[i] for i in sorted_ids]
        logprobs = [logprobs[i] for i in sorted_ids]
    return choice_strs, logprobs

def get_candidate_pitches_and_logprobs(pitch_prob_roll, max_num_pitch = 12, max_num_choice = 16):
    support_set = get_support_set(pitch_prob_roll, max_num_pitch)
    possible_choices = powerset(support_set)[1:]
    choice_strs = []
    logprobs = []
    str2logprob = {}
    for choice in possible_choices:
        choice_str = chord_tuple_list_to_string(choice, baseline = 0)
        choice_strs.append(choice_str)

        choice_arr = np.array(choice)
        unchoiced = ranked_list_diff(support_set, choice)
        if len(unchoiced)>0:
            unchoiced_arr = np.array(unchoiced)
            logprobs.append(get_roll_weight(pitch_prob_roll[choice_arr]) + get_roll_weight(1 - pitch_prob_roll[unchoiced]))
        else:
            logprobs.append(get_roll_weight(pitch_prob_roll[choice_arr]))

    choice_strs, logprobs = remove_less_possible_choices(choice_strs, logprobs, max_num_choice)
    n_choice = len(choice_strs)
    for i_choice in range(len(choice_strs)):
        str2logprob[choice_strs[i_choice]] = logprobs[i_choice]
    return str2logprob

''' To explain the concept:
e.g., prev_pitch_roll = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ]
      transition_prior= [.1,.1,.2,.3,.2,.1,.1,.1,.1,.2,.3,.2,.1,.1]
'''
def get_transition_prior(prev_pitch_roll, choice_str, 
    base_prob = 0.001, stay_prob = 0.5, max_prob = 0.75,
    strong_relate_prob = 0.2, strong_relate_intervals = [1,2,3,4,5,7,12],
    weak_relate_prob = 0.15, weak_relate_intervals = [6,8,9,11]
):
    pitch_num = len(prev_pitch_roll)
    all_pitches = list(range(pitch_num))
    if len(choice_str) > 0: 
        choice_arr = np.array(chord_string_to_list(choice_str))
        unchoiced_arr = np.array(ranked_list_diff(all_pitches, choice_arr))
        transition_prior_roll = np.zeros(pitch_num)
        
        prev_pitches = np.nonzero(prev_pitch_roll)[0]
        transition_prior_roll[prev_pitches] = stay_prob
        for prev_p in prev_pitches:
            for i in strong_relate_intervals:
                if prev_p - i >= 0:
                    transition_prior_roll[prev_p - i] += strong_relate_prob
                if prev_p + i < pitch_num:
                    transition_prior_roll[prev_p + i] += strong_relate_prob
            for i in weak_relate_intervals:
                if prev_p - i >= 0:
                    transition_prior_roll[prev_p - i] += weak_relate_prob
                if prev_p + i < pitch_num:
                    transition_prior_roll[prev_p + i] += weak_relate_prob
        
        transition_prior_roll[transition_prior_roll == 0] = base_prob
        transition_prior_roll = np.minimum(transition_prior_roll, max_prob)
        prior = np.prod(transition_prior_roll[choice_arr])*np.prod((1-transition_prior_roll)[unchoiced_arr])
    else:
        transition_prior_roll = np.zeros(pitch_num)
        prior = 1
    return transition_prior_roll, prior

def get_transition_posterior(prev_pitch_roll, choice_str, note_transition_dict, 
    n_prior = 100
):
    prev_pitch_roll_str = chord_tuple_list_to_string(np.nonzero(prev_pitch_roll)[0].tolist())
    if prev_pitch_roll_str in note_transition_dict:
        transition_choices = note_transition_dict[prev_pitch_roll_str]
        num_all = sum(transition_choices.values())
        if choice_str in transition_choices:
            posterior = transition_choices[choice_str] / num_all
        else:
            posterior = 0
        w_posterior = num_all * 1.0 / (num_all + n_prior)
    else:
        posterior = 0
        w_posterior = 0
    
    _, prior = get_transition_prior(prev_pitch_roll, choice_str)
    return w_posterior * posterior + (1-w_posterior) * prior

def single_chord_hmm_viterbi_decoding(pitch_prob_rolls, note_transition_dict, verbose = False):
    n_chords, num_pitches = pitch_prob_rolls.shape
    candidate_chord_strings = [None]*n_chords
    temp_values = [None]*n_chords
    temp_cursers = [None]*n_chords
    
    # get support sets and transmit probabilities
    if verbose:
        print("-= Getting support sets and transmit probabilities =-")
    for i_chord in range(n_chords):
        transmit_logprobs_dict = get_candidate_pitches_and_logprobs(pitch_prob_rolls[i_chord]) # format: {"44,45": -0.3, ...}
        for pitches_str in list(transmit_logprobs_dict.keys()):
            pitches_list = chord_string_to_list(pitches_str)
            if not pitches_sanity_check(pitches_list):
                transmit_logprobs_dict.pop(pitches_str)
        
        n_candidates = len(transmit_logprobs_dict.keys())
        if n_candidates > 0:
            candidate_chord_strings[i_chord] = list(transmit_logprobs_dict.keys())
            temp_values[i_chord] = np.array(list(transmit_logprobs_dict.values()))
            temp_cursers[i_chord] = [None]*n_candidates
        else:
            candidate_chord_strings[i_chord] = [""]
            temp_values[i_chord] = np.array([1])
            temp_cursers[i_chord] = [0]
        
        if verbose and (i_chord < 5 or i_chord == n_chords - 1):
            if i_chord == n_chords - 1:
                print("...")
            print("chord number",i_chord,"support sets:", candidate_chord_strings[i_chord][0], "; ...;", candidate_chord_strings[i_chord][-1], "; total num: ", n_candidates)
    
    if verbose:
        print("-= Implementing Viterbi =-")
    for i_chord in range(1, n_chords):
        candidates_prev = candidate_chord_strings[i_chord-1]
        candidates_this = candidate_chord_strings[i_chord]
        n_candidates_prev = len(candidates_prev)
        n_candidates_this = len(candidates_this)
        
        # acquire previous valid pitch rolls
        candidates_prev_lists = [chord_string_to_list(chord_str) for chord_str in candidates_prev]
        candidates_prev_rolls = [np.zeros(num_pitches) for _ in range(n_candidates_prev)]
        for i_prev in range(n_candidates_prev):
            candidates_prev_rolls[i_prev][np.nonzero(candidates_prev_lists[i_prev])[0]] = 1
        
        # acquire transition probabilities
        transition_probs = np.zeros((n_candidates_prev, n_candidates_this))
        for i_this in range(n_candidates_this):
            for i_prev in range(n_candidates_prev):
                transition_probs[i_prev, i_this] = get_transition_posterior(
                    candidates_prev_rolls[i_prev], candidates_this[i_this], note_transition_dict
                )
        transition_logprobs = np.log2(transition_probs+0.00001)
        
        # implement viterbi
        for i_this in range(n_candidates_this):
            probs_under_study = temp_values[i_chord-1][:] + transition_logprobs[:, i_this]
            i_prev_max = np.argmax(probs_under_study)
            temp_values[i_chord][i_this] += probs_under_study[i_prev_max]
            temp_cursers[i_chord][i_this] = i_prev_max
            
        if verbose and (i_chord < 5 or i_chord == n_chords - 1):
            if i_chord == n_chords - 1:
                print("...")
            print("chord number",i_chord,"Viterbi temp values:", temp_values[i_chord])


    opt_decisions = [None]*n_chords
    # backtrack
    last_opt_id = np.argmax(temp_values[-1])
    opt_decisions[-1] = candidate_chord_strings[-1][last_opt_id]
    for i_chord in reversed(range(1, n_chords)):
        last_opt_id = temp_cursers[i_chord][last_opt_id]
        opt_decisions[i_chord-1] = candidate_chord_strings[i_chord-1][last_opt_id]
    
    return opt_decisions

def str_decisions_to_pitch_rolls(decisions, num_pitches = 88):
    n_chords = len(decisions)
    pitch_lists = [chord_string_to_list(chord_str) for chord_str in decisions]
    pitch_rolls = np.zeros((n_chords, num_pitches))
    for i_chord in range(n_chords):
        if len(decisions[i_chord]) > 0:
            pitch_rolls[i_chord][np.array(pitch_lists[i_chord])] = 1
    return pitch_rolls



