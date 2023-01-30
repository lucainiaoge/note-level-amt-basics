import os
import sys
from pydantic import BaseModel
from typing import List

import numpy as np
from scipy import signal
from scipy import stats
from scipy.stats import beta
import json
import h5py
import soundfile as sf
import mido
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from IPython.display import Audio, display

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(directory.parent)
from algorithm.onset_offset_detection import get_onset_offsets

# data formats
from hyperparameters import SUPPORTED_MIDI_TYPES, SUPPORTED_AUDIO_TYPES
# constants on midi
from hyperparameters import MAESTRO_BPM, MAESTRO_TEMPO, MAESTRO_TICK_PER_BEAT, MAESTRO_TICK_PER_SEC
# constants on audio
from hyperparameters import SOURCE_SAMPLE_RATE, PROJECT_SAMPLE_RATE, N_BINS
# constants of dataset
from hyperparameters import NUM_CHORD_PER_H5, MIN_AUDIO_LEN, MAX_AUDIO_LEN, FIXED_LEN

def play_audio(waveform, sample_rate):
    shape  = waveform.shape
    if len(shape) == 2:
        num_channels = np.min(shape)
        num_frames = np.max(shape)
    elif len(shape) == 1:
        num_channels = 1
        num_frames = len(waveform)
    else:
        raise ValueError("Waveform array with more than 2 dimensions are not supported.")
    if num_channels == 1:
        if len(shape) == 1:
            display(Audio(waveform, rate=sample_rate))
        elif len(shape) == 2:
            display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")

def resample_audio(x, fs_input, fs_output, axis=0):
    return signal.resample(x, int(len(x)*fs_output/fs_input), axis=axis)
        
def read_audio(audio_path, sample_rate = None, combine_channels = True):
    x, fs = sf.read(audio_path)
    if sample_rate is None or sample_rate == fs:
        sample_rate = fs
    else:
        x = resample_audio(x, fs, sample_rate)
    if combine_channels and len(x.shape)==2:
        x = x.sum(axis = 1)
    return x, sample_rate

def read_audio_section(filename, start_time, stop_time):
    track = sf.SoundFile(filename)

    can_seek = track.seekable() # True
    if not can_seek:
        raise ValueError("Not compatible with seeking")

    sr = track.samplerate
    start_frame = int(sr * start_time)
    frames_to_read = int(sr * (stop_time - start_time))
    track.seek(start_frame)
    audio_section = track.read(frames_to_read)
    return audio_section, sr

def read_audio_to_1_channel_simple(audio_path):
    waveform, fs = sf.read(audio_path)
    if len(waveform.shape)==2:
        waveform = waveform[:,0]
    return waveform, fs

def extract_as_clip(input_filename, output_filename, start_time, stop_time):
    audio_extract, sr = read_audio_section(input_filename, start_time, stop_time)
    sf.write(output_filename, audio_extract, sr)
    return

def read_txt_by_line(txt_file_path, dtype=float):
    places = []
    # Open the file and read the content in a list
    with open(txt_file_path, 'r') as filehandle:
        for line in filehandle:
            curr_place = line[:-1]
            places.append(dtype(curr_place))
    return places


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.
    return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)
    

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


def maestro_get_tempo(mido_midi_obj):
    for msg in mido_midi_obj.tracks[0]:
        if msg.type == "set_tempo":
            return msg.tempo
    return None

def maestro_get_tick_per_second(mido_midi_obj):
    tempo = maestro_get_tempo(mido_midi_obj)
    bps = 1.0 * 1e6 / tempo
    return round(mido_midi_obj.ticks_per_beat * bps)

def maestro_remove_cc(mido_midi_obj):
    bonus_dur = 0
    N_msgs = len(mido_midi_obj.tracks[1])
    i = 0
    while i < N_msgs:
        msg = mido_midi_obj.tracks[1][i]
        if msg.type == "control_change" and (msg.control==64 or msg.control==67) and i < N_msgs - 2:
            bonus_dur += msg.time
            mido_midi_obj.tracks[1].pop(i)
            N_msgs -= 1
        else:
            mido_midi_obj.tracks[1][i].time += bonus_dur
            if msg.type == "control_change":
                msg.value = 0
            bonus_dur = 0
            i += 1
    return mido_midi_obj

def maestro_batch_remove_cc_and_save(midi_dir, save_dir):
    assert midi_dir!=save_dir, "set source directory different from saving directory."
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dir_list = os.listdir(midi_dir)
    for filename in dir_list:
        if filename[-4:] == ".mid" or filename[-5:] == ".midi":
            filepath = os.path.join(midi_dir, filename)
            outpath = os.path.join(save_dir, filename)
            this_midi = mido.MidiFile(filepath)
            this_midi = maestro_remove_cc(this_midi)
            this_midi.save(outpath)

class NoteInt(BaseModel):
    pitch: int
    velocity: int
    onset: int
    offset: int = 0

class ConcurrentNotesInt(BaseModel):
    pitches: List[int]
    velocities: List[int]
    onset: int
    offset: int = 0


def maestro_midi_to_note_list(mido_midi_obj):
    note_list = []
    pitch_buf = {}
    time = 0
    N_msgs = len(mido_midi_obj.tracks[1])
    for i in range(N_msgs):
        msg = mido_midi_obj.tracks[1][i]
        time += msg.time
        if msg.type == "note_on" and msg.velocity > 0 and msg.note > 0:
            pitch_buf[msg.note] = NoteInt(pitch = msg.note, velocity = msg.velocity, onset = time)
        elif (msg.type == "note_on" and msg.velocity == 0) or msg.type == "note_off":
            note_finished = pitch_buf.pop(msg.note, None)
            if note_finished is not None:
                note_finished.offset = time
                note_list.append(note_finished)
    return note_list

def remove_short_notes(note_list, min_len = 20):
    N_notes = len(note_list)
    i = 0
    while i < N_notes:
        dur = note_list[i].offset - note_list[i].onset
        if dur < min_len:
            note_list.pop(i)
            N_notes -= 1
        else:
            i += 1
    return note_list
    

# pitch_range: the number of pitches considered
# lowest_pitch: the lowest pitch (MIDI number) in the considered data; for piano, the A0 note (lowest piano key) has midi
# returns: np array in shape (max_len, pitch_range)
def maestro_note_list_to_piano_roll(note_list, pitch_range = 88, lowest_pitch = 21):
    max_len = note_list[-1].offset
    piano_roll = np.zeros((max_len, pitch_range))
    for note in note_list:
        piano_roll[note.onset:note.offset, note.pitch-lowest_pitch] = note.velocity
    return piano_roll

def maestro_onset_offset_pitch_list_to_piano_roll(onset_list, offset_list, pitch_list, pitch_range = 88, lowest_pitch = 21):
    max_len = offset_list[-1]
    piano_roll = np.zeros((max_len, pitch_range))
    for i_note in range(len(onset_list)):
        piano_roll[onset_list[i_note]:offset_list[i_note], pitch_list[i_note]-lowest_pitch] = 1
    return piano_roll

# auxiliary function detecting consecutive notes in the same pitch
def rectify_conved_pitch_vec(conved_pitch_vec):
    non_zero_ids = np.nonzero(conved_pitch_vec)[0]
    
    prev_is_positive = None
    prev_i = 0
    for i in non_zero_ids:
        this_is_positive = conved_pitch_vec[i]>0
        if prev_is_positive is not None:
            if (prev_is_positive and this_is_positive):
                conved_pitch_vec[i] = 1
                conved_pitch_vec[i-1] = -1
            elif (not prev_is_positive and not this_is_positive):
                conved_pitch_vec[prev_i] = 1
                conved_pitch_vec[prev_i-1] = -1
        prev_is_positive = this_is_positive
        prev_i = i
    return conved_pitch_vec

def unify_onset_offset_ids(onset_ids, offset_ids, unify_range = 2):
    n_notes = len(onset_ids)
    for i_note in range(n_notes-1):
        offset = offset_ids[i_note]
        next_onset = onset_ids[i_note+1]
        if next_onset - offset <= unify_range and next_onset > offset:
            offset_ids[i_note] = next_onset
    return onset_ids, offset_ids

def get_onset_offset_ids_from_piano_roll(piano_roll):
    max_len, pitch_range = piano_roll.shape
    
    piano_roll = np.append(piano_roll, np.zeros((1,pitch_range)), axis=0)
    kernel_onset = np.array([1,-1])
    onset_array_all = np.empty((0,), dtype=int)
    offset_array_all = np.empty((0,), dtype=int)
    for p in range(pitch_range):
        # convolve flipps the kernel, unlike correlation operation
        conved_pitch_vec = np.convolve(piano_roll[:,p], kernel_onset, mode='same')
        conved_pitch_vec = rectify_conved_pitch_vec(conved_pitch_vec)
        onset_vec_p = conved_pitch_vec>0
        offset_vec_p = conved_pitch_vec<0
        
        onset_ids_p = np.nonzero(onset_vec_p)[0]
        
        if len(onset_ids_p) > 0:
            offset_ids_p = np.nonzero(offset_vec_p)[0]
            assert len(onset_ids_p) == len(offset_ids_p), "onset offset not same length at pitch "+str(p)+":"+str(len(onset_ids_p))+","+str(len(offset_ids_p))
            assert np.all(onset_ids_p<=offset_ids_p), "onset offset value not matching at pitch "+str(p)

            onset_array_all = np.append(onset_array_all, onset_ids_p)
            offset_array_all = np.append(offset_array_all, offset_ids_p)
            
    onset_order = np.argsort(onset_array_all)
    onset_ids = onset_array_all[onset_order]
    offset_ids = offset_array_all[onset_order]
    onset_ids, offset_ids = unify_onset_offset_ids(onset_ids, offset_ids, unify_range = 2)
    piano_roll = piano_roll[:-1,:] # this is because of the extension operation in get_onset_offset_ids_from_piano_roll()
    return onset_ids, offset_ids

def maestro_piano_roll_to_note_list(piano_roll, lowest_pitch = 21, amp2vel_mapping = None):
    max_len, pitch_range = piano_roll.shape
    
    piano_roll = np.append(piano_roll, np.zeros((1,pitch_range)), axis=0)
    kernel_onset = np.array([1,-1])
    note_list = []
    onset_array_all = np.empty((0,), dtype=int)
    for p in range(pitch_range):
        # convolve flipps the kernel, unlike correlation operation
        conved_pitch_vec = np.convolve(piano_roll[:,p], kernel_onset, mode='same')
        conved_pitch_vec = rectify_conved_pitch_vec(conved_pitch_vec)
        onset_vec_p = conved_pitch_vec>0
        offset_vec_p = conved_pitch_vec<0
        
        onset_ids_p = np.nonzero(onset_vec_p)[0]
        
        if len(onset_ids_p) > 0:
            offset_ids_p = np.nonzero(offset_vec_p)[0]
            assert len(onset_ids_p) == len(offset_ids_p), "onset offset not same length at pitch "+str(p)+":"+str(len(onset_ids_p))+","+str(len(offset_ids_p))
            assert np.all(onset_ids_p<=offset_ids_p), "onset offset value not matching at pitch "+str(p)
            
            onset_array_all = np.append(onset_array_all, onset_ids_p)
            for i_id in range(len(onset_ids_p)):
                mean_amp = np.mean(piano_roll[onset_ids_p[i_id]:offset_ids_p[i_id], p])
                if amp2vel_mapping is None:
                    velocity = min(int(mean_amp), 127)
                else:
                    velocity = amp2vel_mapping(mean_amp)

                note_list.append(NoteInt(
                    pitch = p+lowest_pitch, 
                    velocity = velocity,
                    onset = onset_ids_p[i_id], 
                    offset = offset_ids_p[i_id]
                ))
    
    onset_order = np.argsort(onset_array_all)
    piano_roll = piano_roll[:-1,:]
    return [note_list[i] for i in onset_order]

def decompose_note_list(note_list):
    n_note = len(note_list)
    pitch_list = [None]*n_note
    velocity_list = [None]*n_note
    onset_list = [None]*n_note
    offset_list = [None]*n_note
    for i_note in range(n_note):
        pitch_list[i_note] = note_list[i_note].pitch
        velocity_list[i_note] = note_list[i_note].velocity
        onset_list[i_note] = note_list[i_note].onset
        offset_list[i_note] = note_list[i_note].offset
    return pitch_list, velocity_list, onset_list, offset_list

def get_concurrent_notes_from_piano_roll(piano_roll, lowest_pitch = 21, min_len = 2):
    onset_ids, offset_ids = get_onset_offset_ids_from_piano_roll(piano_roll)
    concurrent_notes_list, concurrent_notes_onsets, concurrent_notes_offsets = get_concurrent_notes_from_piano_roll_given_onset_offsets(
        piano_roll, onset_ids, offset_ids, lowest_pitch = lowest_pitch, min_len = min_len
    )
    return concurrent_notes_list, concurrent_notes_onsets, concurrent_notes_offsets

def get_pitch_activation_roll_mode(pitch_activation_roll):
    length, pitch_num = pitch_activation_roll.shape
    if length == 1 or length == 2:
        return pitch_activation_roll[0,:]
    else:
        return np.squeeze(stats.mode(pitch_activation_roll, axis=0)[0])

def get_concurrent_notes_from_piano_roll_given_onset_offsets(piano_roll, onset_ids, offset_ids, lowest_pitch = 21, min_len = 2):
    max_tick, pitch_range = piano_roll.shape
    n_segments = len(onset_ids)
    assert len(onset_ids) == len(offset_ids), "Input onset and offset array should be of the same length"

    raw_merged_event_times = np.concatenate((onset_ids, offset_ids))
    raw_merged_event_times = np.sort(raw_merged_event_times)
    merged_event_times = [0]
    prev_time = 0
    for i in range(len(raw_merged_event_times)):
        this_time = raw_merged_event_times[i]
        if this_time > max_tick:
            break
        elif (this_time-prev_time)>=min_len or prev_time <= min_len:
            merged_event_times.append(this_time)
            prev_time = this_time
    
    concurrent_notes_list = []
    concurrent_notes_onsets = []
    concurrent_notes_offsets = []
    prev_time = 0
    for i in range(len(merged_event_times)):
        this_time = merged_event_times[i]
        this_pitch_roll_segment = (piano_roll[prev_time:this_time] > 0) * 1
        pitch_activation = get_pitch_activation_roll_mode(this_pitch_roll_segment)
        
        if (pitch_activation>0).any():
            pitches = np.nonzero(pitch_activation)[0] + lowest_pitch
            velocities = np.mean(piano_roll[prev_time:this_time,pitches-lowest_pitch], axis=0)

            concurrent_notes_list.append(ConcurrentNotesInt(
                pitches = pitches.tolist(),
                velocities = velocities.tolist(),
                onset = prev_time,
                offset = this_time
            ))
            concurrent_notes_onsets.append(prev_time)
            concurrent_notes_offsets.append(this_time)
        
        prev_time = this_time
    
    return concurrent_notes_list, concurrent_notes_onsets, concurrent_notes_offsets

def maestro_concurrent_notes_to_piano_roll(concurrent_notes_list, pitch_range = 88, lowest_pitch = 21):
    max_len = concurrent_notes_list[-1].offset
    piano_roll = np.zeros((max_len, pitch_range))
    for notes in concurrent_notes_list:
        for pitch, velocity in zip(notes.pitches, notes.velocities):
            piano_roll[notes.onset:notes.offset, pitch-lowest_pitch] = velocity
    return piano_roll

# onsets, offsets are in sec; pitch_rolls, velocity_rolls are list of vectors
def pitch_velocity_rolls_onsets_offsets_to_piano_roll(pitch_rolls, velocity_rolls, onsets, offsets, 
    tick_per_sec = MAESTRO_TICK_PER_SEC, relative_time = False
):
    n_chords, num_pitches = pitch_rolls.shape
    assert pitch_rolls.shape == velocity_rolls.shape, "pitch and velocity roll should have the same size"
    
    global_onsets = (tick_per_sec * onsets).astype(int)
    global_offsets = (tick_per_sec * offsets).astype(int)
    masked_velocity_rolls = np.copy(velocity_rolls[:])
    masked_velocity_rolls[pitch_rolls == 0] = 0
    
    if not relative_time:
        piano_roll = np.zeros((global_offsets[-1], num_pitches))
        for i_chord in range(n_chords):
            piano_roll[global_onsets[i_chord]:global_offsets[i_chord], :] = masked_velocity_rolls[i_chord]
    else:
        bias =  global_onsets[0]
        piano_roll = np.zeros((global_offsets[-1]-bias, num_pitches))
        for i_chord in range(n_chords):
            piano_roll[global_onsets[i_chord]-bias:global_offsets[i_chord]-bias, :] = masked_velocity_rolls[i_chord]
    return piano_roll

def rectify_chord_onset_offset_seq(chord_vecs, onset_ids_midi, offset_ids_midi, min_len = 20, max_gap = 10):
    assert len(chord_vecs) == len(onset_ids_midi)
    assert len(onset_ids_midi) == len(offset_ids_midi)
    onset_ids_midi = np.array(onset_ids_midi)
    offset_ids_midi = np.array(offset_ids_midi)
    durs = offset_ids_midi - onset_ids_midi
    gaps = onset_ids_midi[1:] - offset_ids_midi[0:-1]
    for i_vec in range(1, len(chord_vecs)):
        this_dur = durs[i_vec]
        this_gap = gaps[i_vec - 1]
        if this_dur <= min_len and this_gap <= max_gap:
            chord_vecs[i_vec] = chord_vecs[i_vec - 1]
    return chord_vecs
        

def rectify_piano_roll(piano_roll, reduce_map = "max", max_gap = 40):
    dtype = piano_roll.dtype
    max_tick, num_pitch = piano_roll.shape

    rectified_piano_roll = np.zeros_like(piano_roll, dtype = dtype)
    binary_roll = piano_roll.astype(bool) * 1
    # fill up the gaps
    for p in range(num_pitch):
        nonzero_t = np.nonzero(binary_roll[:,p])[0]
        prev_note_len = 1
        prev_gap = 100000
        if len(nonzero_t) > 0:
            prev_i_midi = nonzero_t[0]
            for i in range(1,len(nonzero_t)):
                this_i_midi = nonzero_t[i]
                if this_i_midi == prev_i_midi + 1:
                    prev_note_len += 1
                elif this_i_midi - prev_i_midi <= max_gap:
                    if prev_note_len > max_gap or prev_note_len < max_gap/2:
                        binary_roll[prev_i_midi:this_i_midi,p] = 1
                        prev_note_len += (this_i_midi - prev_i_midi)
                    else:
                        prev_note_len = 1
                        prev_gap = this_i_midi - prev_i_midi
                else:
                    prev_note_len = 1
                    prev_gap = this_i_midi - prev_i_midi
                prev_i_midi = this_i_midi

    # average up the note velocities
    note_list = maestro_piano_roll_to_note_list(binary_roll, lowest_pitch = 0)
    for note in note_list:
        onset = note.onset
        offset = note.offset
        pitch = note.pitch
        if reduce_map == "max":
            velocity = np.max(piano_roll[onset:offset, pitch]).astype(dtype)
        else:
            velocity = np.mean(piano_roll[onset:offset, pitch]).astype(dtype)
        rectified_piano_roll[onset:offset, pitch] = velocity
    return rectified_piano_roll

class NoteOnOffEvent():
    def __init__(self, onset: int, pitch: int, velocity: int):
        self.onset = onset
        self.pitch = pitch
        self.velocity = velocity # velocity = 0 means offset

    def __lt__(self, other):
        return self.onset < other.onset

def maestro_note_list_to_midi(note_list, save_path):
    event_list = []
    for note in note_list:
        event_list.append(NoteOnOffEvent(note.onset, note.pitch, note.velocity))
        event_list.append(NoteOnOffEvent(note.offset, note.pitch, 0))

    event_list.sort()

    # create midi
    midi_obj = mido.MidiFile()
    meta_track = mido.MidiTrack()
    note_track = mido.MidiTrack()

    meta_track.append(mido.MetaMessage("set_tempo", tempo=MAESTRO_TEMPO, time=0))
    meta_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    meta_track.append(mido.MetaMessage("end_of_track", time=1))
    note_track.append(mido.Message("program_change", channel=0, program=0, time=0))

    prev_onset = 0
    for event in event_list:
        this_onset = event.onset
        dur = this_onset - prev_onset
        note_track.append(mido.Message("note_on", note=event.pitch, velocity=event.velocity, time=dur))
        prev_onset = this_onset

    note_track.append(mido.MetaMessage("end_of_track", time=1))

    midi_obj.tracks.append(meta_track)
    midi_obj.tracks.append(note_track)

    midi_obj.save(save_path)
    return midi_obj

# event_list: [(pitch, velocity (0 for offset), duration (in midi ticks))]
def maestro_event_list_to_midi(event_list, save_path):
    # create midi
    midi_obj = mido.MidiFile()
    meta_track = mido.MidiTrack()
    note_track = mido.MidiTrack()

    meta_track.append(mido.MetaMessage("set_tempo", tempo=MAESTRO_TEMPO, time=0))
    meta_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    meta_track.append(mido.MetaMessage("end_of_track", time=1))
    note_track.append(mido.Message("program_change", channel=0, program=0, time=0))
    
    for event in event_list:
        note_track.append(mido.Message("note_on", note=event[0], velocity=event[1], time=event[2]))

    note_track.append(mido.MetaMessage("end_of_track", time=1))

    midi_obj.tracks.append(meta_track)
    midi_obj.tracks.append(note_track)

    midi_obj.save(save_path)
    return midi_obj

def velocity2amp(velocity, dB_range = 40):
    r = np.power(10,  (dB_range / 20))
    b = 127 / (126 * np.sqrt( r )) - 1 / 126
    m = (1 - b) / 127
    return 2 * (m * velocity + b) ** 2

def amp2velocity_appox(amp, max_amp = 2.0):
    return np.clip(np.round(127 * np.log10(9 * np.maximum(amp, 0) / max_amp + 1)), 0, 127)

def maestro_piano_roll_to_midi(piano_roll, save_path, lowest_pitch = 21, amp2vel_mapping = None):
    note_list = maestro_piano_roll_to_note_list(piano_roll, lowest_pitch, amp2vel_mapping)
    return maestro_note_list_to_midi(note_list, save_path)

def maestro_read_midi_to_note_list(midi_path, min_len = 20):
    midi_obj = mido.MidiFile(midi_path)
    tick_per_sec = maestro_get_tick_per_second(midi_obj)

    note_list = maestro_midi_to_note_list(midi_obj)
    note_list = remove_short_notes(note_list, min_len = min_len)
    return note_list, tick_per_sec

def maestro_read_midi_to_piano_roll(midi_path, pitch_range = 88, lowest_pitch = 21, min_len = 20):
    note_list, tick_per_sec = maestro_read_midi_to_note_list(midi_path, min_len = min_len)
    piano_roll = maestro_note_list_to_piano_roll(note_list, pitch_range = pitch_range, lowest_pitch = lowest_pitch)
    return piano_roll, tick_per_sec



def chop_audio_by_segments(waveform, sr, concurrent_notes_onsets, concurrent_notes_offsets, tick_per_sec = MAESTRO_TICK_PER_SEC):
    assert len(concurrent_notes_onsets) == len(concurrent_notes_onsets), "onset and offset lists should be in the same size"
    N_test_concurrent_notes = len(concurrent_notes_onsets)
    wav_segments = []
    for i in range(N_test_concurrent_notes):
        start_second = concurrent_notes_onsets[i] / tick_per_sec
        end_second = concurrent_notes_offsets[i] / tick_per_sec
        wav_segments.append(waveform[int(start_second*sr): int(end_second*sr)])

    return wav_segments


def remove_midi_filename_extension(midi_filename):
    assert type(midi_filename)==str, "should input a string"
    for extension in SUPPORTED_MIDI_TYPES:
        if extension in midi_filename:
            if midi_filename[-len(extension):] == extension:
                return midi_filename[:-len(extension)]
    return None

def remove_audio_filename_extension(audio_filename):
    assert type(audio_filename)==str, "should input a string"
    for extension in SUPPORTED_AUDIO_TYPES:
        if extension in audio_filename:
            if audio_filename[-len(extension):] == extension:
                return audio_filename[:-len(extension)]
    return None

def pitch_vec_to_piano_roll(pitches, pitch_range = 88, lowest_pitch = 21):
    pitch_roll = np.zeros(pitch_range, dtype=bool)
    for p in pitches:
        pitch_roll[p-lowest_pitch] = True
    return pitch_roll

def velocity_vec_to_piano_roll(velocities, pitches, pitch_range = 88, lowest_pitch = 21, transform_velocity = True):
    assert len(velocities) == len(pitches), "input velocity vec and pitch vec should have same length."
    pitch_roll = np.zeros(pitch_range, dtype=bool)
    velocity_roll = np.zeros(pitch_range, dtype=np.float32)
    for i in range(len(pitches)):
        p = pitches[i]-lowest_pitch
        pitch_roll[p] = True
        velocity_roll[p] = velocities[i]
        if transform_velocity:
            velocity_roll[p] = velocity_roll[p] / 128 # velocity2amp(velocity_roll[p], dB_range = 40)
    return pitch_roll, velocity_roll

def get_spectrum(x, N_bins = N_BINS, sample_rate = PROJECT_SAMPLE_RATE, mode = "numpy", div_reduce = 10.0):
    if mode == "numpy":
        fft_spectrum = np.abs(np.fft.rfft(x, n=N_bins*2, norm = "ortho"))**2
        return fft_spectrum[:N_bins] / div_reduce
    else:
        fft_spectrum = np.abs(np.fft.rfft(x, norm = "ortho"))**2
        target_spectrum = np.zeros(N_bins)

        fft_freq_grid = np.arange(len(fft_spectrum)) * 1.0 / len(fft_spectrum) * sample_rate / 2
        target_freq_grid = np.arange(N_bins) * 1.0 / N_bins * sample_rate / 2
        boundary = 1.0 / N_bins * sample_rate / 2
        
        i_target_grid = 0
        for i_fft_grid in range(len(fft_spectrum)):
            this_freq = fft_freq_grid[i_fft_grid]
            while abs(target_freq_grid[i_target_grid] - this_freq) > boundary and i_target_grid < N_bins - 1:
                i_target_grid += 1
            target_spectrum[i_target_grid] += fft_spectrum[i_fft_grid]
        return target_spectrum / div_reduce

def clip_or_pad_to_fixed_len(x, fixed_len: int):
    assert fixed_len > 0, "fixed length should be greater than 0"
    if len(x) >= fixed_len:
        x_out = x[:fixed_len]
    else:
        x_out = np.pad(x, (0, fixed_len-len(x)), 'constant', constant_values=(0, 0))
    return x_out

# todo: segment with algorihtm
class PieceDataset(Dataset):
    def __init__(self, midi_dir, audio_dir,
        pitch_range = 88, lowest_pitch = 21, min_len = 20, fix_len = False,
        transform_velocity = True, audio_extension = ".wav", resample_rate = None, time_as_sec = False,
        chop_with_algorithm = False
    ):
        midi_names = os.listdir(midi_dir)
        audio_names = os.listdir(audio_dir)
        piece_names_by_audio = [remove_audio_filename_extension(filename) for filename in audio_names]
        self.piece_names = []
        self.midi_names = []
        for i_file,midi_name in enumerate(midi_names):
            piece_name = remove_midi_filename_extension(midi_name)
            if piece_name is not None and piece_name in piece_names_by_audio:
                self.piece_names.append(piece_name)
                self.midi_names.append(midi_name)

        self.midi_dir = midi_dir
        self.audio_dir = audio_dir
        self.pitch_range = pitch_range
        self.lowest_pitch = lowest_pitch
        self.min_len = min_len
        self.fix_len = fix_len
        self.transform_velocity = transform_velocity
        self.audio_extension = audio_extension
        self.resample_rate = resample_rate
        self.time_as_sec = time_as_sec
        self.chop_with_algorithm = chop_with_algorithm

    def __len__(self):
        return len(self.piece_names)

    def __getitem__(self, idx):
        piece_name = self.piece_names[idx]
        midi_name = self.midi_names[idx]
        audio_name = piece_name + self.audio_extension
        
        midi_path = os.path.join(self.midi_dir, midi_name)
        audio_path = os.path.join(self.audio_dir, audio_name)

        piano_roll, tick_per_sec = maestro_read_midi_to_piano_roll(midi_path, 
            pitch_range = self.pitch_range, lowest_pitch = self.lowest_pitch, min_len = self.min_len
        )
        waveform, source_sample_rate = read_audio_to_1_channel_simple(audio_path)
        
        if not self.chop_with_algorithm:
            concurrent_notes_list, concurrent_notes_onsets, concurrent_notes_offsets = get_concurrent_notes_from_piano_roll(
                piano_roll, lowest_pitch = self.lowest_pitch, min_len = 2
            )
        else:
            onset_ids_audio, offset_ids_audio = get_onset_offsets(waveform, source_sample_rate)
            onset_ids_midi = (onset_ids_audio / source_sample_rate * tick_per_sec).astype(int)
            offset_ids_midi = (offset_ids_audio / source_sample_rate * tick_per_sec).astype(int)
            concurrent_notes_list, concurrent_notes_onsets, concurrent_notes_offsets = get_concurrent_notes_from_piano_roll_given_onset_offsets(
                piano_roll, onset_ids_midi, offset_ids_midi, lowest_pitch = self.lowest_pitch, min_len = 2
            )
            
        wav_segments = chop_audio_by_segments(
            waveform, source_sample_rate, concurrent_notes_onsets, concurrent_notes_offsets, 
            tick_per_sec = tick_per_sec
        )

        velocity_rolls = [None]*len(wav_segments)
        spectrums = [None]*len(wav_segments)
        for i_seg in range(len(wav_segments)):
            spectrums[i_seg] = get_spectrum(wav_segments[i_seg], N_bins = N_BINS, sample_rate = source_sample_rate)
            if self.resample_rate is not None:
                wav_segments[i_seg] = resample_audio(wav_segments[i_seg], source_sample_rate, self.resample_rate, axis=0)
                out_sample_rate = self.resample_rate
            else:
                out_sample_rate = source_sample_rate
            if self.fix_len:
                wav_segments[i_seg] = clip_or_pad_to_fixed_len(wav_segments[i_seg], FIXED_LEN)
            if self.time_as_sec:
                concurrent_notes_onsets[i_seg] = float(concurrent_notes_onsets[i_seg]) / tick_per_sec
                concurrent_notes_offsets[i_seg] = float(concurrent_notes_offsets[i_seg]) / tick_per_sec
            velocities = concurrent_notes_list[i_seg].velocities
            pitches = concurrent_notes_list[i_seg].pitches
            _, velocity_rolls[i_seg] = velocity_vec_to_piano_roll(velocities, pitches, 
                pitch_range = self.pitch_range, lowest_pitch = self.lowest_pitch, transform_velocity = self.transform_velocity
            )
        
        return {
            "name": piece_name,
            "wav_segs": wav_segments,
            "spectrums": spectrums,
            "velocity_rolls": velocity_rolls,
            "onsets": concurrent_notes_onsets,
            "offsets": concurrent_notes_offsets,
            "chords": concurrent_notes_list,
            "sample_rate": out_sample_rate
        }

def maestro_create_single_chord_dataset_json(
    midi_dir, audio_dir, save_dir, 
    min_len = 20, pitch_range = 88, lowest_pitch = 21, 
    audio_extension = ".wav"
):

    piece_dataset = PieceDataset(midi_dir, audio_dir,
        pitch_range = pitch_range, lowest_pitch = lowest_pitch, min_len = min_len, 
        transform_velocity = False, audio_extension = audio_extension
    )
    for i_piece in range(len(piece_dataset)):
        piece_info_dict = piece_dataset[i_piece]
        piece_name = piece_info_dict["name"]
        chord_dataset = []
        for i_seg in range(len(piece_info_dict["wav_segs"])):
            chord_dataset.append({
                "name": piece_info_dict["name"],
                "audio": piece_info_dict["wav_segs"][i_seg].tolist(),
                "onset": int(piece_info_dict["onsets"][i_seg]),
                "offset": int(piece_info_dict["offsets"][i_seg]),
                "pitches": piece_info_dict["chords"][i_seg].pitches,
                "velocities": piece_info_dict["chords"][i_seg].velocities
            })

        with open(os.path.join(save_dir, piece_name+".json"), 'w') as fout:
            json.dump(chord_dataset, fout)
    print("piece "+piece_name+" finished!")

def maestro_create_single_chord_dataset_h5(
    midi_dir, audio_dir, save_dir, 
    min_len = 20, pitch_range = 88, lowest_pitch = 21, 
    audio_extension = ".wav", num_chord_per_h5 = NUM_CHORD_PER_H5
):
    piece_dataset = PieceDataset(midi_dir, audio_dir,
        pitch_range = pitch_range, lowest_pitch = lowest_pitch, min_len = min_len, 
        transform_velocity = False, audio_extension = audio_extension
    )
    
    audio_buf = [np.array([0], dtype=np.int16)]*num_chord_per_h5
    pitches_buf = [np.array([0], dtype=np.int16)]*num_chord_per_h5
    velocities_buf = [np.array([0], dtype=np.int16)]*num_chord_per_h5
    onset_buf = [np.array([0], dtype=np.int16)]*num_chord_per_h5
    offset_buf = [np.array([0], dtype=np.int16)]*num_chord_per_h5
    prev_velocity_roll = [np.array([0], dtype=np.int16)]*num_chord_per_h5
    cum_i_seg = 0
    for i_piece in range(len(piece_dataset)):
        piece_info_dict = piece_dataset[i_piece]
        piece_name = piece_info_dict["name"]
        for i_seg in range(len(piece_info_dict["wav_segs"])):
            i_chord = cum_i_seg % num_chord_per_h5
            if i_chord == 0:
                i_h5 = int(cum_i_seg / num_chord_per_h5)
                packed_hdf5_path = os.path.join(save_dir, str(i_h5)+".h5")
            
            if len(piece_info_dict["wav_segs"][i_seg]) > 50 and len(piece_info_dict["chords"][i_seg].pitches) > 0:
                audio_buf[i_chord] = float32_to_int16(piece_info_dict["wav_segs"][i_seg])
                pitches_buf[i_chord] = np.array(piece_info_dict["chords"][i_seg].pitches, dtype=np.int16)
                velocities_buf[i_chord] = np.array(piece_info_dict["chords"][i_seg].velocities, dtype=np.int16)
                onset_buf[i_chord] = np.array(piece_info_dict["onsets"][i_seg], dtype=np.int16)
                offset_buf[i_chord] = np.array(piece_info_dict["offsets"][i_seg], dtype=np.int16)
                if i_chord > 0:
                    prev_velocity_roll[i_chord] = np.zeros(pitch_range, dtype=np.int16)
                    prev_velocity_roll[i_chord][pitches_buf[i_chord-1]-lowest_pitch] = velocities_buf[i_chord-1]
                else:
                    prev_velocity_roll[i_chord] = np.zeros(pitch_range, dtype=np.int16)
                cum_i_seg += 1
                if i_chord == num_chord_per_h5 - 1:
                    with h5py.File(packed_hdf5_path, 'w') as hf:
                        for i_chord_out in range(num_chord_per_h5):
                            hf.create_dataset(name="audio_"+str(i_chord_out), data=audio_buf[i_chord_out], dtype=np.int16)
                            hf.create_dataset(name="pitches_"+str(i_chord_out), data=pitches_buf[i_chord_out], dtype=np.int16)
                            hf.create_dataset(name="velocities_"+str(i_chord_out), data=velocities_buf[i_chord_out], dtype=np.int16)
                            hf.create_dataset(name="onset_"+str(i_chord_out), data=onset_buf[i_chord_out], dtype=np.int16)
                            hf.create_dataset(name="offset_"+str(i_chord_out), data=offset_buf[i_chord_out], dtype=np.int16)
                            hf.create_dataset(name="prevchord_"+str(i_chord_out), data=prev_velocity_roll[i_chord_out], dtype=np.int16)
        print("h5 file number "+str(i_h5), "piece "+piece_name+" finished!")

def maestro_parse_single_chord_h5(hdf5_path, i_chord,
    source_sample_rate = SOURCE_SAMPLE_RATE, prj_sample_rate = PROJECT_SAMPLE_RATE, 
    pitch_range = 88, lowest_pitch = 21, 
    tick_per_sec = MAESTRO_TICK_PER_SEC, transform_velocity = True,
    num_chord_per_h5 = NUM_CHORD_PER_H5
):
    assert i_chord < num_chord_per_h5, "query chord id is larger than the h5 chord size"
    with h5py.File(hdf5_path, 'r') as hf:
        audio = int16_to_float32(hf["audio_"+str(i_chord)][:])
        pitches = hf["pitches_"+str(i_chord)][:]
        velocities = hf["velocities_"+str(i_chord)][:]
        onset = np.array(hf["onset_"+str(i_chord)], dtype=float) / tick_per_sec
        offset = np.array(hf["offset_"+str(i_chord)], dtype=float) / tick_per_sec
        pitch_roll, velocity_roll = velocity_vec_to_piano_roll(velocities, pitches, 
            pitch_range = pitch_range, lowest_pitch = lowest_pitch, transform_velocity = transform_velocity
        )
        prevchord = (hf["prevchord_"+str(i_chord)][:] * 1.0 / 128).astype(float)
        return {
            "audio": resample_audio(audio, source_sample_rate, prj_sample_rate, axis=0),
            "pitch_roll": pitch_roll,
            "velocity_roll": velocity_roll,
            "onset": onset,
            "offset": offset,
            "prevchord": prevchord
        }

def audio_len_to_pitch_repeat_vec(pitch_repeat:int, data_len, min_len = 410, max_len = 1600):
    # if data_len is smaller than average, then skew left; otherwise, skew right
    param0 = 5
    ave_len = (min_len + max_len) / 2
    half_len = (max_len - min_len) / 2
    len_shift = data_len - ave_len
    param_shift_factor = max(1 - np.abs(len_shift)/half_len, 0.1)
    if len_shift > 0:
        a = param0
        b = param0*param_shift_factor
    else:
        a = param0*param_shift_factor
        b = param0
    
    sample_space = np.linspace(start=0.05, stop=0.95, num=pitch_repeat)
    dist_vec = beta.pdf(sample_space, a, b)
    return dist_vec / dist_vec.sum()