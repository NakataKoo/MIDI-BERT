# MidiBERT-Piano

## Introduction
This is not official repository for the paper, [MidiBERT-Piano: Large-scale Pre-training for Symbolic Music Understanding](https://arxiv.org/pdf/2107.05223.pdf).

With this repository, you can
* pre-train a MidiBERT-Piano with your customized pre-trained dataset
* fine-tune & evaluate on 4 downstream tasks
* extract melody (mid to mid) using pre-trained MidiBERT-Piano

All the datasets employed in this work are publicly available.

## Installation
* Python3
* Install generally used packages for MidiBERT-Piano:
```python
git clone https://github.com/wazenmai/MIDI-BERT.git
cd MIDI-BERT
pip install -r requirements.txt
```

研究室のA40サーバー（CUDA 12.1）では、以下でも上手くいった(torch=2.2.0, CUDA=12.1,)

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage
Please see `scripts` folder, which includes bash file for
* prepare data
* pretrain
* finetune
* evaluation
* melody extraction

You may need to change the folder/file name or any config settings you prefer.


## Repo Structure
```
Data/
└── Dataset/       
    └── pop909/       
    └── .../
└── CP_data/
    └── pop909_train.npy
    └── *.npy

data_creation/
└── preprocess_pop909/
└── prepare_data/       # convert midi to CP_data 
    └── dict/           # CP dictionary 

melody_extraction/
└── skyline/
└── midibert/

MidiBERT/
└── *py

```

## More
For more details on 
* data preparation, please go to `data_creation` and follow Readme
* MidiBERT pretraining, finetuning, evaluation, please go to `MidiBERT` and follow Readme.
* skyline, please go to `melody_extraction/skyline` and follow Readme.
* pianoroll figure generation, please go to `melody_extraction/pianoroll` and follow Readme. We also provide clearer pianoroll pictures of the paper.
* listening to melody extraction results, please go to `melody_extraction/audio` and read Readme for more details.

Note that Baseline (LSTM) and code in remi versions are removed for cleaness.  But you could find them in `main` branch.

## Citation

[Midi-BERT Official repo](https://github.com/wazenmai/MIDI-BERT/tree/CP).
