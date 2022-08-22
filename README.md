# Perceived Information Revisited
This repository is for a paper published in issue 4, TCHES 2022, entitled "Perceived Information Revisited: New Metrics to Evaluate Success Rate of Side-Channel Attacks."

This repository contains the traces obtained form masked AES hardware based on threshold implementation (TI) presented in COSADE 2017, and some of source code for an attack on this hardware.

## Quick Start Guide
1. Clone this repository to get the source code for the experiment.
```
git clone https://github.com/ECSIS-lab/perceived_information_revisited.git
```

2. Download the zip archive of dataset from ほちゃらら, and unzip it.
3. Execute ./scripts/run.sh
```
/bin/zsh ./scripts/run.sh
```

## Repository structure
The structure of this repository is shown below:
```
.
├── LICENSE
├── README.md
├── model.h5
├── scripts
│   └── run.sh
└── srcs
    ├── key_est_nonshare.py
    ├── load_data.py
    └── model.py
```

#### model.h5
A hdf5 file which contains the pre-trained model parameters for an attack on masked AES implementation.

#### scripts/run.sh
A zsh script for a demonstration of an experimental attack.

#### srcs/key\_est\_nonshare.py
A python script to recover the secret key using the pre-trained model.

#### srcs/load\_data.py
A python script to load the dataset.

#### srcs/model.py
A python script that contains a structure of CNN model used in the experiment.
