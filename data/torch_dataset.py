import json
import os
import sys
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset
import mido

from .data_utils import get_spectrum, maestro_parse_single_chord_h5, clip_or_pad_to_fixed_len
from .data_utils import MAESTRO_TICK_PER_SEC, SOURCE_SAMPLE_RATE, PROJECT_SAMPLE_RATE
from .data_utils import NUM_CHORD_PER_H5, MIN_AUDIO_LEN, MAX_AUDIO_LEN, FIXED_LEN
from .data_utils import audio_len_to_pitch_repeat_vec

class SingleChordDataset(Dataset):
    def __init__(self, hdf5s_dir, 
        source_sample_rate = SOURCE_SAMPLE_RATE, prj_sample_rate = PROJECT_SAMPLE_RATE, 
        pitch_range = 88, lowest_pitch = 21, tick_per_sec = MAESTRO_TICK_PER_SEC, 
        pitch_repeat:int = 1, transform_velocity = True,
        random_len = False, fix_len = False, insert_spectrum = True
    ):
        self.hdf5s_dir = hdf5s_dir
        self.hdf5s_files = os.listdir(hdf5s_dir)
        self.source_sample_rate = source_sample_rate
        self.prj_sample_rate = prj_sample_rate
        self.pitch_range = pitch_range
        self.lowest_pitch = lowest_pitch
        self.pitch_repeat = pitch_repeat
        self.tick_per_sec = tick_per_sec
        self.transform_velocity = transform_velocity
        self.random_len = random_len
        self.fix_len = fix_len
        self.insert_spectrum = insert_spectrum

    def __len__(self):
        # return len(self.hdf5s_files)
        return (len(self.hdf5s_files) - 1)*NUM_CHORD_PER_H5

    def __getitem__(self, idx):
        i_h5 = int(idx / NUM_CHORD_PER_H5)
        hdf5_path = os.path.join(self.hdf5s_dir, self.hdf5s_files[i_h5])
        i_chord = idx % NUM_CHORD_PER_H5
        h5_dict = maestro_parse_single_chord_h5(hdf5_path, i_chord, 
            source_sample_rate = self.source_sample_rate, prj_sample_rate = self.prj_sample_rate, 
            pitch_range = self.pitch_range, lowest_pitch = self.lowest_pitch, 
            tick_per_sec = self.tick_per_sec, transform_velocity = self.transform_velocity
        )
        if self.insert_spectrum:
            h5_dict["spectrum"] = get_spectrum(h5_dict["audio"])
        if self.random_len:
            min_len = min(MIN_AUDIO_LEN-1, len(h5_dict["audio"])-1)
            max_len = min(MAX_AUDIO_LEN, len(h5_dict["audio"]))
            wave_len = np.random.randint(min_len, max_len)
            h5_dict["audio"] = h5_dict["audio"][:wave_len]
        else:
            wave_len = len(h5_dict["audio"])
        if self.fix_len:
            h5_dict["audio"] = clip_or_pad_to_fixed_len(h5_dict["audio"], FIXED_LEN)
        if self.pitch_repeat > 1:
            if self.fix_len:
                THIS_MAX_LEN = FIXED_LEN
            else:
                THIS_MAX_LEN = MAX_AUDIO_LEN
            # in shape (pitch_repeat,)
            repeat_vec = audio_len_to_pitch_repeat_vec(self.pitch_repeat, wave_len, min_len = MIN_AUDIO_LEN, max_len = THIS_MAX_LEN)
            repeated_vel_roll = np.repeat(h5_dict["velocity_roll"][:,np.newaxis], self.pitch_repeat, axis=1)
            repeated_vel_roll = repeated_vel_roll * repeat_vec
            h5_dict["velocity_roll_repeat"] = repeated_vel_roll.flatten()
        else:
            h5_dict["velocity_roll_repeat"] = h5_dict["velocity_roll"]
        return h5_dict

class TransmitModelDataset(Dataset):
    def __init__(self, json_dir, data_len = 8, hop_len = 4):
        self.json_dir = json_dir
        self.data_len = data_len
        self.hop_len = 4
        json_names = os.listdir(json_dir)
        self.useable_names = []
        
        # those list components are pieces
        self.gt_velocity_rolls_list = []
        self.est_velocity_rolls_list = []
        self.onsets_list = []
        self.offsets_list = []
        self.piece_len_list = []
        
        for i_file,json_name in enumerate(json_names):
            extension = json_name[-5:]
            if extension == ".json":
                self.useable_names.append(json_name)
            with open(os.path.join(self.json_dir,json_name), 'r') as fin:
                transcribed_data_dict = json.load(fin)
                self.gt_velocity_rolls_list.append(np.array(transcribed_data_dict["gt_velocity_rolls"]))
                self.est_velocity_rolls_list.append(np.array(transcribed_data_dict["est_velocity_rolls"]))
                self.onsets_list.append(np.array(transcribed_data_dict["onsets"]))
                self.offsets_list.append(np.array(transcribed_data_dict["offsets"]))
                self.piece_len_list.append(len(transcribed_data_dict["onsets"]))
                
        self.max_load_list = [int((len_piece-data_len)/hop_len + 1) for len_piece in self.piece_len_list]
        self.idx_list = np.cumsum(self.max_load_list)
        self.idx_list = np.insert(self.idx_list, 0, 0)
        
    def __len__(self):
        return self.idx_list[-1]

    def __getitem__(self, idx):
        selected_piece_id = 0
        for piece_id in range(len(self.idx_list)):
            if idx >= self.idx_list[piece_id] and idx < self.idx_list[piece_id+1]:
                selected_piece_id = piece_id
        piece_hop_num = idx - self.idx_list[selected_piece_id]
        piece_start_id = piece_hop_num * self.hop_len
        piece_end_id = piece_start_id + self.data_len
        # print(selected_piece_id, self.piece_len_list[selected_piece_id], piece_start_id, piece_end_id)
        
        return {
            "gt_velocity_rolls": self.gt_velocity_rolls_list[selected_piece_id][piece_start_id:piece_end_id],
            "est_velocity_rolls": self.est_velocity_rolls_list[selected_piece_id][piece_start_id:piece_end_id],
            "onsets": self.onsets_list[selected_piece_id][piece_start_id:piece_end_id],
            "offsets": self.offsets_list[selected_piece_id][piece_start_id:piece_end_id]
        }

def collate_fn_single_chord(list_data_dict, 
    min_audio_len = MIN_AUDIO_LEN, max_audio_len = MAX_AUDIO_LEN
):
    """Collate input and target of segments to a mini-batch.

    Args:
      list_data_dict: e.g. [
        {"audio": (T1,), "velocity_roll": (pitch_num), ...}, 
        {"audio": (T2,), "velocity_roll": (pitch_num), ...}, 
        ...]

    Returns:
      np_data_dict: e.g. {
        "audio": (batch_size, Tmax)
        "velocity_roll": (batch_size, pitch_num), 
        ...}
    """
    batch_size = len(list_data_dict)
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = [data_dict[key] for data_dict in list_data_dict]

    if "audio" in np_data_dict:
        audio_len_list = [len(audio) for audio in np_data_dict["audio"]]
        max_len = max(audio_len_list)
        max_len = max([max_len, MIN_AUDIO_LEN])
        max_len = min([max_len, MAX_AUDIO_LEN])
        for i_batch in range(batch_size):
            this_len = audio_len_list[i_batch]
            if this_len <= max_len:
                np_data_dict["audio"][i_batch] = np.pad(
                    np_data_dict["audio"][i_batch], 
                    (0, max_len - audio_len_list[i_batch]), 
                    'constant', constant_values=(0, 0)
                )
            else:
                np_data_dict["audio"][i_batch] = np_data_dict["audio"][i_batch][:max_len]

    for key in np_data_dict.keys():
        np_data_dict[key] = np.array(np_data_dict[key])
    
    return np_data_dict

def collate_fn_transduction(list_data_dict):
    """Collate input and target of segments to a mini-batch.

    Args:
      list_data_dict: [{
            "gt_velocity_rolls": (data_len, pitch_num),
            "est_velocity_rolls": (data_len, pitch_num),
            "onsets": (data_len,),
            "offsets": (data_len,)
      }]

    Returns:
      np_data_dict: e.g. {
        "gt_velocity_rolls": (batch_size, data_len, pitch_num),
        "est_velocity_rolls": (batch_size, data_len, pitch_num),
        "onsets": (batch_size, data_len),
        "offsets": (batch_size, data_len)
      }
    """
    batch_size = len(list_data_dict)
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = [data_dict[key] for data_dict in list_data_dict]
    for key in np_data_dict.keys():
        np_data_dict[key] = np.array(np_data_dict[key])
    
    return np_data_dict