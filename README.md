![generated-sample-sheet-music](https://github.com/anatrini/Performance-RNN-PyTorch/blob/master/imgs/score.png)

# Performance RNN - PyTorch

This repository contains a PyTorch implementation of Performance RNN, a model inspired by the work of Ian Simon and Sageev Oore on "Performance RNN: Generating Music with Expressive Timing and Dynamics," as presented in the Magenta Blog in 2017. [https://magenta.tensorflow.org/performance-rnn](https://magenta.tensorflow.org/performance-rnn).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Purpose

The implementation of this model is part of the educational activities conducted during the Artificial Models for Music Creativity class at Hochschule für Musik und Theater Hamburg, held during the Winter Semester 2023/2024. For more context and resources related to the class, please refer to the class repository https://github.com/anatrini/Artificial-Models-Music-Creativity

This model is not implemented in the official way!

Noteworthy edits from the original fork includes:

- PyTorch 2 Implementation: This version of Performance RNN has been upgraded to PyTorch 2, harnessing the latest advancements in the framework.

- GPU Calculation Support: The model seamlessly supports GPU calculations, allowing for faster and more efficient processing, especially on Mac OS platforms.

- Configurable batch number, early stopping, model's performance visualization, logging and hyperparameters optimization routine. New config file has been introduced, empowering users to easily set batch numbers and enable early stopping during model training.


## Installation

To install and run this project, it is recommended to set up a virtual environment using tools like Conda or Poetry with Python 3.10. For instance using Conda, follow these steps after cloning the repository and navigating to the root folder:

1. Create a new Conda environment with Python 3.10:

    `conda create --name <environment_name> python==3.10`

2. Activate the newly created environment:

    `conda activate <myenv>`

3. Install PyTorch 2.3.0 from the PyTorch Nightly channel:

    `conda install -c pytorch-nightly pytorch=2.3.0`

4. Install the required packages listed in the requirements.txt file:

    `pip install -r requirements.txt`

If you want to use this repository as a Python module, replace the last command with the following:

    `pip install -e .`

This will install the package in editable mode, which means changes to the source code will be immediately effective without the need for reinstallation.

Ensure that `pip` is available within your environment and make sure to activate the environment before installing the requirements.

Once the installtion is complete, you can import the module in your Python script as follows:

    ```python import performance_rnn_torch as rnn```


## Directory Structure

```
.
├── dataset/
│   ├── midi/
│   │   ├── dataset1/
│   │   │   └── *.mid
│   │   └── dataset2/
│   │       └── *.mid
│   ├── processed/
│   │   └── dataset1/
│   │       └── *.data (preprocess.py)
│   └── scripts/
│       └── *.sh (dataset download scripts)
│       └── maestro_parser.py (maestro dataset composer extractor)
├── output/
│   └── *.mid (generate.py)
├── save/
│   └── *.sess (train.py)
└── runs/ (tensorboard logdir)
```


## Instructions

- Download datasets

You can use one of the shell scripts to download a dataset for training the model if you are using a UNIX machine. 

    `cd dataset/`
    `bash scripts/<script_name>_scraper.sh midi/<folder_name>`

If you already have the maestro dataset from https://magenta.tensorflow.org/datasets/maestro#v300, you can use the provided Python script to selectively copy MIDI files corresponding to a specific composer.

    `cd script/`
    `python maestro_parser.py -c <composer_name> -s <output_dir>, -m <path to .csv maestro's dataset content>`

- Preprocessing

Once you got the midi files they must be preprocessed.

    `python preprocess -m dataset/midi/<folder_name_to_preprocess> -s dataset/processed/<folder_name_for_preprocessed_files> -w <number_of_workers>`

By default, the number of workers is set to 0. This configuration implies that the preprocessing steps will be executed on a single thread. To leverage the full processing power of your machine, consider increasing this number to run the preprocessing steps concurrently on multiple threads. Adjust the worker count based on the number of processors available on your machine.

- Training

To train the model using .data files located in dataset/processed/mydata and save the model to save/mymodel.sess every 3 minutes (time interval is expressed in seconds), you can refer to the following example:

    `python train.py -S save/mymodel.sess -d dataset/processed/mydata -i 180`

For a comprehensive list of available options, you can run:

    `python train.py -h`

or 

    `python train.py --help`

![training-figure](https://github.com/anatrini/Performance-RNN-PyTorch/blob/master/imgs/tensorboard.png)

- Hyperparameters optimization
Before initiating the model training process, it's advisable to identify the optimal settings for both model architecture and training hyperparameters. This can be achieved through the following step:

    `python hyperparameters.py -S save/mymodel.sess -d dataset/processed/mydata`

- Generating

To generate midi files you can refer to the following examples.
Generate with control sequence from test.data and model from save/test.sess:

    `python generate.py -s save/test.sess -c test.data`

Generate with pitch histogram and note density (C major scale):

    `python generate.py -s save/test.sess -l 1000 -c '1,0,1,0,1,1,0,1,0,1,0,1;3'`

Uniform generation, chromatic scale:

    `python generate.py -s save/test.sess -l 1000 -c ';3'`

Using control sequence from processed data

    `python generate.py -s save/test.sess -c dataset/processed/some/processed.data`

Without any conditioning, thus no control

    `python generate.py -s save/test.sess -l 1000`

For a comprehensive list of available options, you can run:

    `python generate.py -h`

Please check below the synthax of conditioning options.
    
![generated-sample-1](https://github.com/anatrini/Performance-RNN-PyTorch/blob/master/imgs/piano_roll.png)


## Generated Samples

- A sample on C Major Scale [[MIDI](https://drive.google.com/open?id=1mZtkpsu1yA8oOkE_1b2jyFsvCW70FiKU), [MP3](https://drive.google.com/open?id=1UqyJ9e58AOimFeY1xoCPyedTz-g2fUxv)]
    - control option: `-c '1,0,1,0,1,1,0,1,0,1,0,1;4'`
- A sample on C Minor Scale [[MIDI](https://drive.google.com/open?id=1lIVCIT7INuTa-HKrgPzewrgCbgwCRRa1), [MP3](https://drive.google.com/open?id=1pVg3Mg2pSq8VHJRJrgNUZybpsErjzpjF)]
    - control option: `-c '1,0,1,1,0,1,0,1,1,0,0,1;4'`
- A sample on C Major Pentatonic Scale [[MIDI](https://drive.google.com/open?id=16uRwyntgYTzSmaxhp06kUbThDm8W_vVE), [MP3](https://drive.google.com/open?id=1LSbeVqXKAPrNPCPcjy6FVwUuVo7FxYji)]
    - control option: `-c '5,0,4,0,4,1,0,5,0,4,0,1;3'`
- A sample on C Minor Pentatonic Scale [[MIDI](https://drive.google.com/open?id=1zeMHNu37U6byhT-s63EIro8nL6VkUi8u), [MP3](https://drive.google.com/open?id=1asP1z6u1n3PRSysSnvkt-SabpTgT-_x5)]
    - control option: `-c '5,0,1,4,0,4,0,5,1,0,4,0;3'`


## Pretrained Model

- [ecomp.sess](https://drive.google.com/open?id=1daT6XRQUTS6AQ5jyRPqzowXia-zVqg6m)
    - default configuration
    - dataset: [International Piano-e-Competition, recorded MIDI files](http://www.piano-e-competition.com/)    
- [ecomp_w500.sess](https://drive.google.com/open?id=1jf5j2cWppXVeSXhTuiNfAFEyWFIaNZ6f)
    - window_size: 500
    - control_ratio: 0.7
    - dataset: [International Piano-e-Competition, recorded MIDI files](http://www.piano-e-competition.com/)