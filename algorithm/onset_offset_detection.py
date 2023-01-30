import madmom
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter

from hyperparameters import *

def separate_segment_ids_into_onset_offset(segment_ids, spectrogram, 
    power_thres_factor = POWER_THRES_FACTOR, power_quantile = POWER_QUANTILE
):
    assert len(spectrogram) >= max(segment_ids), "input power vector length should be greater than the largest id"
    power_mean = spectrogram[spectrogram >= spectrogram.mean()].mean()
    power_thres = power_mean * power_thres_factor
    onset_ids = segment_ids[:-1]
    offset_ids = segment_ids[1:]
    
    N_ids = len(segment_ids) - 1
    i_id = 0
    while i_id < N_ids:
        start = onset_ids[i_id]
        end = offset_ids[i_id]
        segment_mean = np.quantile(spectrogram[start:end].flatten(), POWER_QUANTILE)
        # print(power_thres, segment_mean)
        if segment_mean < power_thres:
            # print(onset_ids[i_id], offset_ids[i_id], ", deleted")
            onset_ids = np.delete(onset_ids, i_id)
            offset_ids = np.delete(offset_ids, i_id)
            N_ids -= 1
        else:
            i_id += 1
    return onset_ids, offset_ids
    

def get_onset_offsets(audio, fs, 
    superflux = False,
    stft_window_len_sec = STFT_WINDOW_LEN_SEC,
    stft_hop_sec = STFT_HOP_SEC,
    num_bands = NUM_BANDS_MADMOM_SPEC,
    min_hight = MIN_ONSET_OFFSET_HIGHT,
    power_thres_factor = POWER_THRES_FACTOR,
    power_quantile = POWER_QUANTILE
):
    window_size = int(fs*stft_window_len_sec)
    hop_size= int(fs*stft_hop_sec)
    
    temp = madmom.audio.signal.Signal(audio, sample_rate=fs)
    temp = madmom.audio.signal.FramedSignal(temp, frame_size=window_size, hop_size=hop_size)
    temp = madmom.audio.stft.STFT(temp)
    temp = madmom.audio.spectrogram.Spectrogram(temp)
    temp = madmom.audio.spectrogram.FilteredSpectrogram(temp, num_bands=num_bands)
    temp = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(temp, num_bands=num_bands)
    
    if superflux:
        onset_offset_salience = madmom.features.onsets.spectral_flux(temp)
    else:
        onset_offset_salience = maximum_filter(temp, size=(1,3))
        onset_offset_salience = np.abs(onset_offset_salience[:-1] - onset_offset_salience[1:])
        onset_offset_salience = np.sum(onset_offset_salience, axis=1)
        onset_offset_salience = np.pad(onset_offset_salience, [(0,1)], mode='constant', constant_values=0)
    
    onset_offset_salience = onset_offset_salience / onset_offset_salience.max()
    
    min_frameid_distance = int(MIN_ONSET_OFFSET_TIME_DIFF_SEC * fs / hop_size)
    segment_ids, _ = find_peaks(
        onset_offset_salience, 
        height = min_hight, 
        threshold = None, 
        distance = min_frameid_distance, 
        prominence = None, 
        width = None, 
        wlen = None, rel_height = 0.5, plateau_size = None
    )
    
    onset_ids, offset_ids = separate_segment_ids_into_onset_offset(
        segment_ids, temp, power_thres_factor = power_thres_factor, power_quantile = power_quantile
    )
    onset_ids_audio = onset_ids * hop_size
    offset_ids_audio = offset_ids * hop_size
    return onset_ids_audio, offset_ids_audio
