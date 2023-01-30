The original MAESTRO midi dataset should be placed in midi folder. Please download that from https://magenta.tensorflow.org/datasets/maestro. We are using v2.0.0.

The cc-removed midi files are placed in midi_no_cc folder

synth folder: the audio files synthesized using ReaSynth synthesizer
synth_chopped folder: single-chord dataset of the audio files in synthe folder

fourier_frame_acoustic_model_out folder: estimation-gt pairs for transduction model training
unrolled_mse_model_out folder: estimation-gt pairs for transduction model training

reaper_prjs: the REAPER projects used to process data (e.g., synthesizing the audio)

IMPORTANT: the audio data (synth, synth_chopped) are incomplete. Please check and run the notebooks in order to generate the full dataset.