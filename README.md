# Note Level AMT - Basics

Time: 30/Jan/2023

## Intro

This project is a newly-proposed automatic music transcription (AMT) method, which is able to achieve AMT directly in note level (audio to midi).

The current project is an elementary experiment on synthesized audio, from MAESTRO dataset (2004 branch of data).

The basic note-level AMT workflow is given in the figures folder.

## Usage

### Playing Around
1. Checkout ```01-data_preparation_and_midi_parsing_tutorial.ipynb``` to get  familiar with MAESTRO data parsing
2. Checkout ```02-test_madmom_onset_segmentation.ipynb``` to know about the onset/offset detection algorithm under use
3. Checkout ```03.1-single_chord_fourier_frame_approx_toy_examples.ipynb``` to know about the single-chord AMT algorithm
4. Checkout ```03.2-single_chord_fourier_frame_approx_full_range.ipynb``` to run the single-chord AMT algorithm on the whole MAESTRO-2004-SYNTH dataset and to evaluate the single-chord AMT models
5. Checkout ```04-train_single_chord_acoustic_MAP.ipynb``` to train an update template matrix in the single-chord AMT model
6. Checkout ```05-transduction_model.ipynb``` to train the transmit-probability-estimation neural net, and to implement the HMM transduction model
7. Checkout ```06-baseline_AMT_model_inference.ipynb``` to run the whole note-level AMT pipeline, and to evaluate the testing pieces


### Data Preparation
Download the synthesized MAESTRO 2004 audio from [https://huggingface.co/datasets/lucainiao/MAESTRO_2004_SYNTH](https://huggingface.co/datasets/lucainiao/MAESTRO_2004_SYNTH). Users can also create this dataset using the REAPER project ```data/reaper_prjs/2004.rpp```. The single-chord dataset can be created by running section "Create single-chord dataset" in notebook ```01-data_preparation_and_midi_parsing_tutorial.ipynb```, given the audio dataset.


### Training
1. Run single-chord acoustic model: open ```03.2-single_chord_fourier_frame_approx_full_range.ipynb``` and run section "Pass acoustic model for all pieces getting estimate-gt pairs for transduction model training"
2. Train transduction model: open ```05-transduction_model.ipynb``` and run section "Train the transmit-probability-estimation neural net"
3. (Optional) Update the template matrix: run ```04-train_single_chord_acoustic_MAP.ipynb```

### Inference
All model checkpoints are saved in ```checkpoints``` folder, and the selection of model checkpoints can be done by modifying ```hyperparameters.py```

- Notebook running: run ```06-baseline_AMT_model_inference.ipynb```

- Terminal running: run ```inference_main.py``` using ```python inference_main.py [audio_path] [result_midi_path] baseline -v``` to transcribe a single piece; run ```evaluate_main.py``` using ```python evaluate_main.py [audio_data_dir] [result_midi_dir] --gt_midi_dir [gt_midi_dir] -v``` to batch-transcribe pieces, and specifying ```gt_midi_dir``` will give evaluation results.