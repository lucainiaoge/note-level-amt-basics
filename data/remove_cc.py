import mido
import os

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
