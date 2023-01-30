import os

year = "2004"

current_dir = __file__
parent_dir = os.path.dirname(current_dir)
dataset_dir = os.path.join(parent_dir, "midi_no_cc", year)
log_dir = os.path.join(parent_dir, "synth")
files = os.listdir(dataset_dir)
N_piece = len(files)
RPR_ShowConsoleMsg(str(N_piece)+"\n")

fxname = "ReaSynth (Cockos)"

prj = RPR_EnumProjects(-1, 0, 0)
prj_len = RPR_GetProjectLength(prj) # in seconds
piece_len_list = []
piece_name_list = []
piece_id_start = 0
piece_id_end = min(N_piece, 200)
for i_piece in range(piece_id_start,piece_id_end):
    test_piece = os.path.join(dataset_dir, files[i_piece])
    piece_name_list.append(files[i_piece])

    RPR_InsertMedia(test_piece, 0)
    prj_len_new = RPR_GetProjectLength(prj)
    piece_len = prj_len_new - prj_len
    piece_len_list.append(piece_len)
    prj_len = prj_len_new


    with open(os.path.join(log_dir,f"audio_len_{year}_{piece_id_start}_{piece_id_end}.txt"), 'w') as f:
        for s in piece_len_list:
            f.write(str(s) + '\n')
    with open(os.path.join(log_dir,f"audio_name_{year}_{piece_id_start}_{piece_id_end}.txt"), 'w') as f:
        for s in piece_name_list:
            f.write(str(s) + '\n')

    if i_piece == 0:
        tr = RPR_GetSelectedTrack(0, 0)
        RPR_TrackFX_AddByName(tr, fxname, False, -1000)

    item = RPR_GetMediaItem(0, 0)
    take = RPR_GetActiveTake(item)

    pass
    RPR_MIDI_SelectAll(take, True)

    pass
    # to delete all CC information (takes a lot of time!)
    '''
    tmp = 0
    while tmp >= 0:
        # if idx % 100 == 0:
        #     RPR_ShowConsoleMsg(str(idx)+"\n")
        RPR_MIDI_DeleteCC(take, 0)
        # RPR_MIDI_SetCC(take, pedal_ccid, True, True, 0, 0, 0, 0, 0, True)
        tmp = RPR_MIDI_EnumSelCC(take, 0)
    '''

    pass
    # to print number of notes
    '''
    evt_idx = 0
    for _ in range(100000):

        tmp = RPR_MIDI_EnumSelCC(take, evt_idx)
        if tmp < 0:
            RPR_ShowConsoleMsg(str(evt_idx)+"\n")
            break
        else:
            evt_idx = tmp
            if evt_idx % 100 == 0:
                RPR_ShowConsoleMsg(str(evt_idx)+"\n")
    '''

