import os
import numpy as np

# on general constants
PITCH_NUM = 88
LOWEST_PITCH = 21 # the midi number of the lowest key of piano

# on general note frequency
F_A4 = 440.00
C0_to_A4 = 12*4 + 9
A4_to_C9 = 3 + 12*4
FREQUENCY_FACTORS = 2 ** (np.arange(-C0_to_A4, A4_to_C9) / 12)
PITCH_FREQUENCY_GRID = F_A4 * FREQUENCY_FACTORS

FREQUENCY_FACTORS_FINER = 2 ** (np.arange(-C0_to_A4, A4_to_C9, 0.25) / 12)
PITCH_FREQUENCY_GRID_FINER = F_A4 * FREQUENCY_FACTORS_FINER

UNIFORM_FREQUENCY_GRID = np.arange(50, 8000, 50)
HYBRID_FREQUENCY_GRID = np.concatenate((PITCH_FREQUENCY_GRID[:4*17], UNIFORM_FREQUENCY_GRID[15:]))
HYBRID_FREQUENCY_GRID_FINER = np.concatenate((PITCH_FREQUENCY_GRID_FINER[:8*17-1], UNIFORM_FREQUENCY_GRID[15:]))

# on onset offset detection
POWER_THRES_FACTOR = 0.1
POWER_QUANTILE = 0.4
STFT_WINDOW_LEN_SEC = 0.036 # 0.027
STFT_HOP_SEC = 0.01
NUM_BANDS_MADMOM_SPEC = 12
MIN_ONSET_OFFSET_HIGHT = 0.1 # in [0,1.0]
MIN_ONSET_OFFSET_TIME_DIFF_SEC = 0.036 # 0.025 # make sure MIN_ONSET_OFFSET_TIME_DIFF_SEC > 1.5 * STFT_WINDOW_LEN_SEC (this is an empirical rule)

# commonly-used paths
TEMPLATE_DIR = os.path.join("data","synth_chopped")
TEMPLATE_DATA_NAME = "all_notes_synth.json"
TEMPLATE_PATH = os.path.join(TEMPLATE_DIR,TEMPLATE_DATA_NAME)

ACOUSTIC_NN_CKPT_PATH = os.path.join("checkpoints", "UnrolledMSENet_100000_03-01-2023-08-23-51.pth")
# old ckpts: TransductionFC_490000_18-01-2023-14-40-38.pth, TransductionFC_199999_03-12-2022-17-50-34.pth
TRNASDUCTION_NN_ORI_CKPT_PATH = os.path.join("checkpoints", "TransductionFC_300000_19-01-2023-16-12-53.pth") # (for no-updated template opt) 
# old ckpts: TransductionFC_490000_12-01-2023-23-12-05.pth
TRNASDUCTION_NN_TEMPLATE_UPDATED_CKPT_PATH = os.path.join("checkpoints", "TransductionFC_300000_20-01-2023-15-31-38.pth") # (for updated template opt)
# old ckpts: TransductionFC_490000_14-01-2023-04-49-13.pth
TRNASDUCTION_NN_NUV_CKPT_PATH = os.path.join("checkpoints", "TransductionFC_300000_21-01-2023-05-39-52.pth") # (for no-updated template opt)

TRNASDUCTION_NN_CKPT_PATH = TRNASDUCTION_NN_ORI_CKPT_PATH

# (for NUV opt)
TRNASDUCTION_NN_FOR_E2E_ACOUSTIC_MODEL_CKPT_PATH = os.path.join("checkpoints", "TransductionFC_300000_06-01-2023-17-19-20.pth")
CISOID_A_TEMPLATE_CKPT_PATH = os.path.join("checkpoints", "PitchTemplateMatrix_49500_12-01-2023-05-47-06.npy")

NOTE_TRANSITION_DICT_PATH = os.path.join("data", "note_transition_dict_2004.json")

# data formats
SUPPORTED_MIDI_TYPES = [".mid", ".midi"]
SUPPORTED_AUDIO_TYPES = [".wav", ".mp3"]

# constants on midi
MAESTRO_BPM = 120 # 1 (second) = BPM / 60 (second)
MAESTRO_TEMPO = 500000 # 1 (beat) = TEMPO (micro_second), BPS = 1e6 / TEMPO, BPM = BPS * 60
MAESTRO_TICK_PER_BEAT = 480 # 1 (beat) = TICK_PER_BEAT (midi_tick)
MAESTRO_TICK_PER_SEC = 960 # 1 (second) = MAESTRO_SEC2MIDI (midi_tick), TICK_PER_SEC = TICK_PER_BEAT * BPS

# constants on audio
SOURCE_SAMPLE_RATE = 44100
PROJECT_SAMPLE_RATE = 16000
N_BINS = 1600 # frequency bins when getting spectrum

# constants of dataset
NUM_CHORD_PER_H5 = 1000
MIN_AUDIO_LEN = 10 # 410
MAX_AUDIO_LEN = 3200
FIXED_LEN = 1600

# inference modes
MODEL_TYPE_BASELINE = "baseline"
MODEL_TYPE_UNROLLEDNET = "unrollednet"

test_pieces_2004 = [
    "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_08_Track08_wav",
    "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_10_Track10_wav",
    "MIDI-Unprocessed_SMF_17_R1_2004_03-06_ORIG_MID--AUDIO_20_R2_2004_12_Track12_wav--1",
    "MIDI-Unprocessed_XP_03_R1_2004_01-02_ORIG_MID--AUDIO_03_R1_2004_01_Track01_wav",
    "MIDI-Unprocessed_XP_04_R1_2004_03-05_ORIG_MID--AUDIO_04_R1_2004_06_Track06_wav",
    "MIDI-Unprocessed_XP_08_R1_2004_04-06_ORIG_MID--AUDIO_08_R1_2004_05_Track05_wav--1",
    "MIDI-Unprocessed_XP_11_R1_2004_01-02_ORIG_MID--AUDIO_11_R1_2004_02_Track02_wav",
    "MIDI-Unprocessed_XP_14_R1_2004_04_ORIG_MID--AUDIO_14_R1_2004_04_Track04_wav",
    "MIDI-Unprocessed_XP_14_R1_2004_04_ORIG_MID--AUDIO_14_R1_2004_05_Track05_wav",
    "MIDI-Unprocessed_XP_15_R1_2004_03_ORIG_MID--AUDIO_15_R1_2004_03_Track03_wav",
    "MIDI-Unprocessed_XP_18_R1_2004_01-02_ORIG_MID--AUDIO_18_R1_2004_03_Track03_wav",
    "MIDI-Unprocessed_XP_18_R1_2004_01-02_ORIG_MID--AUDIO_18_R1_2004_05_Track05_wav",
    "MIDI-Unprocessed_XP_19_R1_2004_01-02_ORIG_MID--AUDIO_19_R1_2004_02_Track02_wav"
]
