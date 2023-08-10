import logging
from datetime import datetime
from argparse import ArgumentParser
import os
import random
import re

import numpy as np
import torch

import dataset_loader as dataset_loader
from DKTST import DKTST

def get_args():
    # Setup parser
    parser = ArgumentParser()
    # checkpoint parameters
    parser.add_argument('--chkpnt_dir', type=str, default='./checkpoints')
    parser.add_argument('--continue_chkpnt', type=str, default=None)
    parser.add_argument('--n_epoch', '-e', type=int, default=8000)
    parser.add_argument('--hidden_multi', type=int, default=5) # Hidden dim = In dim * Multiplier
    # training parameters
    parser.add_argument('--dataset', '-d', type=str, default='SQuAD1') # TruthfulQA, SQuAD1, NarrativeQA
    parser.add_argument('--dataset_LLM', '-dl', type=str, default='ChatGPT') # ChatGPT, BloomZ, ChatGLM, Dolly, ChatGPT-turbo, GPT4, StableLM
    parser.add_argument('--s1_type', type=str, default='human') # Type of data (human or machine) for the first sample set
    parser.add_argument('--s2_type', type=str, default='machine') # Type of data (human or machine) for the second sample set
    parser.add_argument('--shuffle', default=False, action='store_true') # Shuffle make sure each pair of answers do not correspond to the same questions
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--batch_size_train', '-btr', type=int, default=2000)
    parser.add_argument('--eval_interval', type=int, default=100) # Evaluation interval
    parser.add_argument('--seed', '-s', type=int, default=1102)
    # validation parameters
    parser.add_argument('--perm_cnt', '-pc', type=int, default=200)
    parser.add_argument('--sig_lvl', '-a', type=float, default=0.05)
    parser.add_argument('--batch_size_test', '-bte', type=int, default=20) # Not used if custom validation procedure is used
    parser.add_argument('--use_custom_test', default=False, action='store_true') # Custom validation that test for a range of batch sizes etc.
    # other parameters
    parser.add_argument('--device', '-dv', type=str, default='auto')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()
    
    # derived parameters
    auto_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = auto_device if args.device == 'auto' else args.device
    # added parameters
    args.dtype = torch.float
    return args

def setup_logs():
    log_level = logging.DEBUG if ARGS.debug else logging.INFO
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=log_level, 
        force=True,
        handlers=[
            logging.FileHandler(f'{CHKPNT_PATH}/training.log', mode='a'),
            logging.StreamHandler()])
    return logging.getLogger(CHKPNT_PATH) # Unique per training

def get_and_create_chkpnt_path():
    if ARGS.continue_chkpnt:
        chkpnt_path = os.path.join(ARGS.chkpnt_dir, ARGS.continue_chkpnt)
    else:
        shuffle_str = 's' if ARGS.shuffle else 'nos'
        chkpnt_path = (
            f'{ARGS.chkpnt_dir}/{ARGS.dataset}_{ARGS.s1_type[0]}{ARGS.s2_type[0]}'
            f'_{shuffle_str}_{ARGS.hidden_multi}_{ARGS.n_epoch}_{ARGS.batch_size_train}_{ARGS.seed}_{START_TIME_STR}')
        if not os.path.exists(chkpnt_path):
            os.makedirs(chkpnt_path)
    return chkpnt_path

def setup_seeds():
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed(ARGS.seed)
    torch.backends.cudnn.deterministic = True

def perform_training():
    # Input validation
    assert ARGS.s1_type in ['human', 'machine'] and ARGS.s2_type in ['human', 'machine'], "S1 and S2 type must be one of: human, machine."
    
    setup_seeds()
    LOGGER.info("============ Training Starts ============\n")

    # Set up dataset
    LOGGER.info(f'Loading dataset {ARGS.dataset}...')
    data = dataset_loader.load(name=ARGS.dataset, detectLLM=ARGS.dataset_LLM)
    assert len(data) % 2 == 0, "Dataset must contain pairs of human and machine answers, i.e. {Human, Machine, Human...}"
    data_human_tr = data['train']['text'][0::2]
    data_machine_tr = data['train']['text'][1::2]
    data_human_te = data['test']['text'][0::2]
    data_machine_te = data['test']['text'][1::2]
    # Random shuffle the dataset so each pair of answers do not correspond to the same questions
    if ARGS.shuffle:
        random.shuffle(data_human_tr)
        random.shuffle(data_machine_tr)
        random.shuffle(data_human_te)
        random.shuffle(data_machine_te)
    
    # Allocate data to S1 and S2
    if ARGS.s1_type == 'human':
        data_s1_tr = data_human_tr
        data_s1_te = data_human_te
    elif ARGS.s1_type == 'machine':
        data_s1_tr = data_machine_tr
        data_s1_te = data_machine_te
    else:
        raise ValueError("Sample data type not recognized")
    
    if ARGS.s2_type == 'human':
        data_s2_tr = data_human_tr
        data_s2_te = data_human_te
    elif ARGS.s2_type == 'machine':
        data_s2_tr = data_machine_tr
        data_s2_te = data_machine_te
    else:
        raise ValueError("Sample data type not recognized")

    # If two sets use the same type, use half of the data of that type for each set so they are disjoint
    if ARGS.s1_type == ARGS.s2_type:
        s = len(data_s1_tr)//2
        data_s1_tr = data_s1_tr[:s]
        data_s2_tr = data_s2_tr[s:s*2]
        s = len(data_s1_te)//2
        data_s1_te = data_s1_te[:s]
        data_s2_te = data_s2_te[s:s*2]

    # Set up DK-TST
    dktst = DKTST(
        latent_size_multi=ARGS.hidden_multi,
        device=ARGS.device,
        dtype=ARGS.dtype,
        logger=LOGGER
    )
    
    # Load correct checkpoint for continue training
    cont_epoch = 0
    if ARGS.continue_chkpnt:
        chkpnt_epoch_max = 0
        chkpnt_names = os.listdir(CHKPNT_PATH)
        for n in chkpnt_names:
            match = re.search(r'model_ep_([0-9]+).pth', n)
            if match and int(match.group(1)) > chkpnt_epoch_max:
                chkpnt_epoch_max = int(match.group(1))
        
        if chkpnt_epoch_max == 0:
            raise Exception('Could not find a valid checkpoint to continue')
        else:
            dktst.load(f"{CHKPNT_PATH}/model_ep_{chkpnt_epoch_max}.pth")
            cont_epoch = chkpnt_epoch_max + 1
            LOGGER.info(f"Continue training from epoch {cont_epoch} for model {ARGS.continue_chkpnt}")
    
    # Basic logging of training parameters
    LOGGER.info(
        f"Training with parameters: \n"
        f"  {CHKPNT_PATH=}\n"
        f"  {ARGS.continue_chkpnt=}\n"
        f"  {ARGS.n_epoch=}\n"
        f"  {ARGS.hidden_multi=}\n"
        f"  {ARGS.dataset=}\n"
        f"  {ARGS.dataset_LLM=}\n"
        f"  {ARGS.s1_type=}\n"
        f"  {ARGS.s2_type=}\n"
        f"  {ARGS.shuffle=}\n"
        f"  {ARGS.learning_rate=}\n"
        f"  {ARGS.batch_size_train=}\n"
        f"  {ARGS.perm_cnt=}\n"
        f"  {ARGS.sig_lvl=}\n"
        f"  Testing batch size={'In Code' if ARGS.use_custom_test else ARGS.batch_size_test}\n"
        f"  {ARGS.use_custom_test=}\n"
        f"  {ARGS.eval_interval=}\n"
        f"  {ARGS.device=}\n"
        f"  {ARGS.seed=}\n"
        )
    
    # Start training
    J_stars, mmd_values, mmd_stds = dktst.train_and_test(
        s1_tr=data_s1_tr,
        s2_tr=data_s2_tr,
        s1_te=data_s1_te,
        s2_te=data_s2_te,
        lr=ARGS.learning_rate,
        n_epoch=ARGS.n_epoch,
        batch_size_tr=ARGS.batch_size_train,
        batch_size_te=ARGS.batch_size_test,
        save_folder=CHKPNT_PATH,
        perm_cnt=ARGS.perm_cnt,
        sig_lvl=ARGS.sig_lvl,
        continue_epoch=cont_epoch,
        use_custom_test=ARGS.use_custom_test,
        eval_inteval=ARGS.eval_interval
    )
    
if __name__ == "__main__":
    START_TIME_STR = datetime.now().strftime("%Y%m%d%H%M%S")
    ARGS = get_args()
    # CHKPNT_PATH = get_and_create_chkpnt_path()
    # LOGGER = setup_logs()
    
    START_TIME_STR = datetime.now().strftime("%Y%m%d%H%M%S")
    ARGS.continue_chkpnt = 'SQuAD1_hm_nos_5_15000_2000_1104_20230810060350'
    ARGS.n_epoch = 20000
    ARGS.seed = 1104
    ARGS.use_custom_test = True
    CHKPNT_PATH = get_and_create_chkpnt_path()
    LOGGER = setup_logs()
    
    perform_training()
