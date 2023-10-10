# DKTST-for-MTD

An adaption of the Deep-kernel Two-sample-test framework for the task of machine text detection, currently only working on distinguishing between human written texts and Chat-GPT written texts.

## Manual Installation on Linux

### Install Miniconda (Instruction below) / Anaconda (Not provided)

```bash
mkdir -p ~/miniconda3 \
&& wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
&& bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
&& rm -rf ~/miniconda3/miniconda.sh \
&& ~/miniconda3/bin/conda init bash 
```

Refresh bash to allow conda to load
```bash
exec bash
```

### Clone this repository to a desired location

```bash
git clone https://github.com/JXKun980/DKTST-for-MTD.git
```

### Create conda environemnt with required packages

Go into repository directory
```bash
cd DKTST-for-MTD
```

If not installed, install build-essential for linux
```bash
apt install build-essential -y
```

Create conda enviornment
```bash
conda env create -f environment.yml
```

### Activate conda environment
```bash
conda acitvate DKTST-for-MTD
```

## Usage

Create a python environemnt with required libraries using venv or conda TODO

Look into the file train.py and test.py for command prompt options TODO
