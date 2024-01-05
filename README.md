![generated-sample-sheet-music](https://user-images.githubusercontent.com/17045050/42017029-3b4f7060-7ae0-11e8-829b-6d6b8b829759.png)

# Performance RNN - PyTorch

This repository contains a PyTorch implementation of Performance RNN, a model inspired by the work of Ian Simon and Sageev Oore on "Performance RNN: Generating Music with Expressive Timing and Dynamics," as presented in the Magenta Blog in 2017. [https://magenta.tensorflow.org/performance-rnn](https://magenta.tensorflow.org/performance-rnn).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Purpose

The implementation of this model is part of the educational activities conducted during the Artificial Models for Music Creativity class at Hochschule für Musik und Theater Hamburg, held during the Winter Semester 2023/2024. For more context and resources related to the class, please refer to the class repository [https://github.com/anatrini/Artificial-Models-Music-Creativity]

This model is not implemented in the official way!

Noteworthy edits from the original fork includes:
- PyTorch 2 Implementation: This version of Performance RNN has been upgraded to PyTorch 2,        harnessing the latest advancements in the framework.

- GPU Calculation Support: The model seamlessly supports GPU calculations, allowing for faster and more efficient processing, especially on Mac OS platforms.

- Configurable Batch Number and Early Stopping: New config file has been introduced, empowering users to easily set batch numbers and enable early stopping during model training.

- Use as a Module: The entire project can be utilized as a module, providing flexibility for integration into larger projects or workflows.

## Installation

Installation

To install and run this project, it is recommended to set up a virtual environment using tools like Conda or Poetry with Python 3.10. Below is an example using Conda:

## Generated Samples

- A sample on C Major Scale [[MIDI](https://drive.google.com/open?id=1mZtkpsu1yA8oOkE_1b2jyFsvCW70FiKU), [MP3](https://drive.google.com/open?id=1UqyJ9e58AOimFeY1xoCPyedTz-g2fUxv)]
    - control option: `-c '1,0,1,0,1,1,0,1,0,1,0,1;4'`
- A sample on C Minor Scale [[MIDI](https://drive.google.com/open?id=1lIVCIT7INuTa-HKrgPzewrgCbgwCRRa1), [MP3](https://drive.google.com/open?id=1pVg3Mg2pSq8VHJRJrgNUZybpsErjzpjF)]
    - control option: `-c '1,0,1,1,0,1,0,1,1,0,0,1;4'`
- A sample on C Major Pentatonic Scale [[MIDI](https://drive.google.com/open?id=16uRwyntgYTzSmaxhp06kUbThDm8W_vVE), [MP3](https://drive.google.com/open?id=1LSbeVqXKAPrNPCPcjy6FVwUuVo7FxYji)]
    - control option: `-c '5,0,4,0,4,1,0,5,0,4,0,1;3'`
- A sample on C Minor Pentatonic Scale [[MIDI](https://drive.google.com/open?id=1zeMHNu37U6byhT-s63EIro8nL6VkUi8u), [MP3](https://drive.google.com/open?id=1asP1z6u1n3PRSysSnvkt-SabpTgT-_x5)]
    - control option: `-c '5,0,1,4,0,4,0,5,1,0,4,0;3'`

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
├── output/
│   └── *.mid (generate.py)
├── save/
│   └── *.sess (train.py)
└── runs/ (tensorboard logdir)
```

## Instructions

- Download datasets

    ```shell
    cd dataset/
    bash scripts/NAME_scraper.sh midi/NAME
    ```

- Preprocessing

    ```shell
    # Preprocess all MIDI files under dataset/midi/NAME
    python preprocess.py dataset/midi/NAME dataset/processed/NAME <no. of workers>
    ```

- Training

    ```shell
    # Train on .data files in dataset/processed/MYDATA, and save to save/myModel.sess every 3 minutes
    python train.py -s save/myModel.sess -d dataset/processed/MYDATA -i 10

    # Or...
    python train.py -s save/myModel.sess -d dataset/processed/MYDATA -p hidden_dim=1024
    python train.py -s save/myModel.sess -d dataset/processed/MYDATA -b 128 -c 0.3
    python train.py -s save/myModel.sess -d dataset/processed/MYDATA -w 100 -S 10
    ```

    ![training-figure](https://user-images.githubusercontent.com/17045050/42135712-7f6e25f4-7d81-11e8-845f-682bd26a3abb.png)


- Generating

    ```shell
    # Generate with control sequence from test.data and model from save/test.sess
    python3 generate.py -s save/test.sess -c test.data

    # Generate with pitch histogram and note density (C major scale)
    python3 generate.py -s save/test.sess -l 1000 -c '1,0,1,0,1,1,0,1,0,1,0,1;3'

    # Or...
    python3 generate.py -s save/test.sess -l 1000 -c ';3' # uniform pitch histogram
    python3 generate.py -s save/test.sess -l 1000 # no control

    # Use control sequence from processed data
    python3 generate.py -s save/test.sess -c dataset/processed/some/processed.data
    ```
    
    ![generated-sample-1](https://user-images.githubusercontent.com/17045050/42017026-37dfd7b2-7ae0-11e8-99a9-75d27510f44b.png)

## Pretrained Model

- [ecomp.sess](https://drive.google.com/open?id=1daT6XRQUTS6AQ5jyRPqzowXia-zVqg6m)
    - default configuration
    - dataset: [International Piano-e-Competition, recorded MIDI files](http://www.piano-e-competition.com/)    
- [ecomp_w500.sess](https://drive.google.com/open?id=1jf5j2cWppXVeSXhTuiNfAFEyWFIaNZ6f)
    - window_size: 500
    - control_ratio: 0.7
    - dataset: [International Piano-e-Competition, recorded MIDI files](http://www.piano-e-competition.com/)

