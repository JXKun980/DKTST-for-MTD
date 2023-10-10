# DKTST-for-MTD

An adaption of the Deep-kernel Two-sample-test framework for the task of machine text detection, currently only working on distinguishing between human written texts and Chat-GPT written texts.

## Manual Installation on Linux

Install Miniconda (Instruction below) / Anaconda (Not provided)
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

Clone this repository to a desired location
```bash
git clone https://github.com/JXKun980/DKTST-for-MTD.git
```

Go into repository directory
```bash
cd DKTST-for-MTD
```

If not installed, install build-essential for linux
```bash
apt install build-essential -y
```

Create conda enviornment (Can take a while)
```bash
conda env create -f environment.yml
```

Activate conda environment
```bash
conda acitvate DKTST-for-MTD
```

Download Dataset files from [Google Drive link](https://drive.google.com/drive/folders/1p4iBeM4r-sUKe8TnS4DcYlxvQagcmola) sourced from [MGTBench](https://github.com/xinleihe/MGTBench), and place them into `DKTST-for-MTD/datasets/` folder.

## Code Structure

```bash
DKTST-for-MTD
├── datasets
│   ├── NarrativeQA_LLMs.csv
│   ├── SQuAD1_LLMs.csv
│   └── TruthfulQA_LLMs.csv
├── models
│   └── model_name
│       ├── checkpoints.pth
│       └── training_config.yml
├── analysis_logs
├── test_logs
├── environment.yml
├── scripts
│   └── save_training_graph_to_png.py
├── external
│   ├── dataset_loader.py
│   └── dktst_utils_HD.py
├── model.py
├── test.py
├── train.py
├── analysis.py
└── util.py
```

- `/datasets`: Dataset files
- `/models`: Trained models, each in its own folder
- `/analysis_logs`: Output files from `analysis.py`
- `/test_logs`: Output files from `test.py`
- `environment.yml`: Conda environment specification file
- `/scripts`: Utility scripts, call from the root directory of the project
- `/external`: Externally sourced scripts (not written by me)
- `model.py`: Model definition
- `test.py`: Testing script, required trained model, output by default saved to `/test_logs`, call `python test.py --help` for more details.
- `train.py`: Training script, models by default saved to `/models`, call `python train.py --help` for more details.
- `analysis.py`: Analysis script, required `.csv` results from the `test.py` script, either print the test results in a condensed table format, or a graphic format. Output by default saved to `/analysis_logs`.
- `util.py`: Shared script

## Usage

### Train

To train a new model, use `train.py` with the following arguments.
```
usage: train.py [-h] [--model_dir str] [--device str] [--debug] [--continue_model str] [--hidden_multi int] [--n_epoch int] --datasets str [--dataset_llm str] [--dataset_train_ratio float] [--s1_type str] [--s2_type str] [--shuffle] [--learning_rate float]
                [--sample_size_train int] [--eval_interval int] [--save_interval int] [--seed int] [--dtype str] [--perm_cnt int] [--sig_lvl float] [--sample_size_test int] [--sample_count_test int]

optional arguments:
  -h, --help            show this help message and exit
  --model_dir str       Directory to save models (default: ./models)
  --device str, -dv str
                        Device to use for training (default: auto)
  --debug               Enable debug mode, which supresses file creations. (default: False)
  --continue_model str  Name of the model folder to continue training. If set, all parameters below are ignored. (default: None)
  --hidden_multi int    Hidden layer size multiple. Hidden dimension = In dimension * Multiple. (default: 3)
  --n_epoch int, -e int
                        Number of epochs to train (default: 3000)
  --datasets str, -d str
                        One or more datasets to split train set from. If more than one used, they are merged together into a single dataset. (default: None)
  --dataset_llm str, -dl str
                        The LLM the machine generated text is extracted from. (default: ChatGPT)
  --dataset_train_ratio float, -dtrr float
                        Ratio of train set to the total dataset (default: 0.8)
  --s1_type str         Type of data (human or machine) for the first sample set (default: human)
  --s2_type str         Type of data (human or machine) for the second sample set (default: machine)
  --shuffle             Enable to shuffle the test set within each distribution to break pair-dependency (default: False)
  --learning_rate float, -lr float
                        Initial learning rate for Adam (default: 0.0005)
  --sample_size_train int, -sstr int
                        Number of samples in each sample set (for the two distributions) for training (default: 20)
  --eval_interval int   Number of epochs between subsequent test for training and validation power (default: 100)
  --save_interval int   Number of epochs between subsequent model checkpoint saves (default: 500)
  --seed int, -s int    Seed for random number generator for training (default: 1103)
  --dtype str           Data type (float or double) for the linear layer (default: float)
  --perm_cnt int, -pc int
                        Number of permutations to use for training and validation power testing (default: 200)
  --sig_lvl float, -a float
                        Significance level for training and validation power testing (default: 0.05)
  --sample_size_test int, -sste int
                        Number of samples in each sample set (for the two distributions) for training and validation power testing (default: 20)
  --sample_count_test int, -scte int
                        Number of testing sample pairs to generate for validation power testing (default: 50)
```

Example
```bash
python train.py --datasets TruthfulQA
```

Or, look into `train.py` and find the `main()` function, follow the commented example to overwrite default arguments, and specify one or more training script to start training without having to set the arguments every time.
In this case, training is simply done by:
```bash
python train.py
```

The trained models are (by default) saved to its own folder in `/models`.

### Test

To test a trained model, use `test.py` with the following arguments.
```
usage: test.py [-h] [--model_dir str] [--device str] [--debug] [--batch_test] [--model_name str] [--chkpnt_epoch int] [--dataset_llm str] [--dataset_train_ratio float] [--shuffle] [--perm_cnt int] [--sig_lvl float]
               [--sample_size int] [--sample_count int] [--seed int] [--test_type str] [--tst_datasets str] [--tst_s1_type str] [--tst_s2_type str] [--sst_user_dataset str] [--sst_fill_dataset str] [--sst_user_type str]
               [--sst_fill_type str] [--sst_true_ratio float] [--sst_strong]

optional arguments:
  -h, --help            show this help message and exit
  --model_dir str       Directory of the trained models. (default: ./models)
  --device str, -dv str
                        Device to run the model on. (default: auto)
  --debug               Enable debug mode, which supresses file creations. (default: False)
  --batch_test          Whether to run in batch test mode. If yes, the following parameters will be ignored, and parameters to test are specified in the perform_batch_test() function. (default: False)
  --model_name str      Name of the model to test. (default: None)
  --chkpnt_epoch int    Epoch count of the checkpoint to load. If not set, the best checkpoint (with file name prefix "model_best_ep_") will be used. (default: None)
  --dataset_llm str, -dl str
                        The LLM the machine generated text is extracted from. (default: ChatGPT)
  --dataset_train_ratio float
                        The proportion of data that is allocated to the training set (remaining is allocated to testing set) (default: 0.8)
  --shuffle             Enable to shuffle the test set within each distribution to break pair-dependency (default: False)
  --perm_cnt int, -pc int
                        Permuatation count for the test (default: 200)
  --sig_lvl float, -a float
                        Significance level for the test (default: 0.05)
  --sample_size int, -ss int
                        The amount of samples in each sample set (same for both sample sets in the pair) (default: 20)
  --sample_count int, -sc int
                        The amount of pairs of sample sets generated for the test (default: 50)
  --seed int, -s int    Seed of the test (default: 1102)
  --test_type str, -tt str
                        Type of test to run (TST or SST) (default: TST)
  --tst_datasets str, -d str
                        Dataset(s) to test. If multiple datasets specified, they will be merged into a single dataset. (default: None)
  --tst_s1_type str     Type of data (human or machine) for the first sample set (default: human)
  --tst_s2_type str     Type of data (human or machine) for the second sample set (default: machine)
  --sst_user_dataset str
                        Dataset used for the user data (default: None)
  --sst_fill_dataset str
                        Dataset used for the filling data (default: None)
  --sst_user_type str   Distribution of data for the user dataset (human or machine) (default: None)
  --sst_fill_type str   Distribution of data for the filling dataset (human or machine) (default: None)
  --sst_true_ratio float
                        Proportion of real data in each sample set that belongs to the user (default: None)
  --sst_strong          Whether to enable strong mode for the single sample test (requiring two different sets) (default: False)
```

Example
Given a trained model at `/models/TruthfulQA_ChatGPT_hm_nos_3_3000_20_1103_5e-05_20230919072909/`, rename the checkpoint `model_ep_1500.pth` to `model_best_ep_1500.pth`, so running `test.py` do not need you to specify the epoch of the checkpoint to test every time.
```bash
python test.py --model_name TruthfulQA_ChatGPT_hm_nos_3_3000_20_1103_5e-05_20230919072909 --test_type TST --tst_datasets TruthfulQA
```

Or, use the `--batch_test` argument, and specify a list of (one or more) values for each parameter in the `perform_batch_test()` function.
In this case, testing is done by simply:
```bash
python test.py --batch_test
```

Test results are (by default) saved to a `test_<start_time_stamp>.log` file and a `test_<start_time_stamp>.csv` file.

### Analysis

Analysis can be performed after a test `.csv` result file has been created.

To run an analysis and print the result in tabular format, use `analysis.py` with the following arguments.
```
usage: analysis.py [-h] --csv_files str --analysis_name {tabular_seed,tabular_shuffle,tabular_linearSize,tabular_sampleSize,tabular_permutationCount,tabular_LLM,tabular_SSTTrueDataRatio,tabular_SSTDataset,graphic_seed,graphic_shuffle,graphic_linearSize,graphic_sampleSize,graphic_sampleSizeAcrossModel,graphic_permutationCount,graphic_permutationCountTiming,graphic_SSTTrueDataRatio} [--test_log_path str] [--output_folder str] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --csv_files str       CSV files to analyze, files are merged into one for analysis. (default: None)
  --analysis_name {tabular_seed,tabular_shuffle,tabular_linearSize,tabular_sampleSize,tabular_permutationCount,tabular_LLM,tabular_SSTTrueDataRatio,tabular_SSTDataset,graphic_seed,graphic_shuffle,graphic_linearSize,graphic_sampleSize,graphic_sampleSizeAcrossModel,graphic_permutationCount,graphic_permutationCountTiming,graphic_SSTTrueDataRatio}
                        Analysis to run, the list of options are specified between {}. (default: None)
  --test_log_path str   Directory to test logs. (default: ./test_logs/)
  --output_folder str   Directory to output analysis logs. (default: ./analysis_logs/)
  --debug               Enable debug model to supress log file creation. (default: False)
```

Example
Given a test result CSV file at `/test_logs/test_20231009082130.csv`, use:
```bash
python analysis.py --csv_files test_20231009082130.csv --analysis_name tabular_shuffle
```

The analysis results are (by default) saved to `/analysis_logs`.

### Utility Scripts

Training graphs are saved to a `events.out.tfevents.*` file in the model's folder, which is a `Tensorboard` log file and can be viewed using `tensorboard --log_dir <model_folder>`.
If instead a `.png` file format is required, the script at `/scripts/save_training_graph_to_png.py` can be used with the following arguments.
```
usage: save_training_graph_to_png.py [-h] [--model_dir MODEL_DIR] [--model_names MODEL_NAMES [MODEL_NAMES ...]]

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Directory where models are saved.
  --model_names MODEL_NAMES [MODEL_NAMES ...]
                        Name of the model folders to save plots for.
```

Example
Given a trained model at `/models/TruthfulQA_ChatGPT_hm_nos_3_3000_20_1103_5e-05_20230919072909/`
```bash
python scripts/save_training_graph_to_png.py --model_names TruthfulQA_ChatGPT_hm_nos_3_3000_20_1103_5e-05_20230919072909
```



