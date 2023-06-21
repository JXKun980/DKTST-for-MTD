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

if __name__ == "__main__":
    # Setup parser
    parser = ArgumentParser()
    # checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--continue_checkpoint', type=str, default=None)
    parser.add_argument('--n_epoch', '-e', type=int, default=1000)
    parser.add_argument('--hidden_multi', type=int, default=3) # Hidden dim = In dim * Multiplier
    # training parameters
    parser.add_argument('--dataset', '-d', type=str, default='SQuAD1') # TruthfulQA, SQuAD1, NarrativeQA
    parser.add_argument('--dataset_LLM', '-dl', type=str, default='ChatGPT') # ChatGPT, BloomZ, ChatGLM, Dolly, ChatGPT-turbo, GPT4, StableLM
    parser.add_argument('--s1_type', type=str, default='human') # Type of data (human or machine) for the first sample set
    parser.add_argument('--s2_type', type=str, default='machine') # Type of data (human or machine) for the second sample set
    parser.add_argument('--shuffle', default=False, action='store_true') # Shuffle make sure each pair of answers do not correspond to the same questions
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.00005)
    parser.add_argument('--batch_size_train', '-btr', type=int, default=2000)
    parser.add_argument('--seed', '-s', type=int, default=1102) # dimension of samples (default value is 10)
    # validation parameters
    parser.add_argument('--perm_count', '-pc', type=int, default=200)
    parser.add_argument('--sig_level', '-a', type=float, default=0.05)
    parser.add_argument('--batch_size_test', '-bte', type=int, default=200) # Not used if custom validation procedure is used
    parser.add_argument('--use_custom_test', default=False, action='store_true') # Custom validation that test for a range of batch sizes etc.
    # other parameters
    parser.add_argument('--device', '-dv', type=str, default='auto')
    parser.add_argument('--debug', default=False, action='store_true')
    
    args = parser.parse_args()
    
    # Setup parameters
    CHKPNT_DIR = args.checkpoint_dir
    CONT_CHKPNT = args.continue_checkpoint
    N_EPOCH = args.n_epoch # number of training epochs
    HIDDEN_MULTI = args.hidden_multi
    
    DATASET = args.dataset
    DATASET_LLM = args.dataset_LLM
    S1_TYPE = args.s1_type
    S2_TYPE = args.s2_type
    SHUFFLE = args.shuffle
    LR = args.learning_rate # default learning rate for MMD-D on HDGM
    BATCH_SIZE_TR = args.batch_size_train
    SEED = args.seed
    
    PERM_CNT = args.perm_count # Amount of permutation during two sample test
    SIG_LVL = args.sig_level
    BATCH_SIZE_TE = args.batch_size_test
    USE_CUSTOM_TEST = args.use_custom_test
    
    auto_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = auto_device if args.device == 'auto' else args.device
    DEBUG = args.debug
    DTYPE = torch.float

    # Input validation
    assert S1_TYPE in ['human', 'machine'] and S1_TYPE in ['human', 'machine'], "S1 and S2 type must be one of: human, machine."
        
    # Model directory, unique per run
    if CONT_CHKPNT:
        CHKPNT_PATH = f'{CHKPNT_DIR}/{CONT_CHKPNT}'
    else:
        start_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        shuffle_str = 's' if SHUFFLE else 'nos'
        CHKPNT_PATH = f'{CHKPNT_DIR}/{DATASET}_{S1_TYPE[0]}{S2_TYPE[0]}_{shuffle_str}_{HIDDEN_MULTI}_{N_EPOCH}_{BATCH_SIZE_TR}_{SEED}_{start_time_str}'
        if not os.path.exists(CHKPNT_PATH):
            os.makedirs(CHKPNT_PATH)
    
    # Setup logs
    log_level = logging.DEBUG if DEBUG else logging.INFO
    logging_path = f'{CHKPNT_PATH}/training.log'
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=log_level, 
        force=True,
        handlers=[
            logging.FileHandler(f'{CHKPNT_PATH}/training.log', mode='a'),
            logging.StreamHandler()])
    logger = logging.getLogger(CHKPNT_PATH) # Unique per training
    
    logger.info("============ Training Starts ============\n")

    # Setup seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Set up dataset
    logger.info(f'Loading dataset {DATASET}...')
    data = dataset_loader.load(name=DATASET, detectLLM=DATASET_LLM)
    assert len(data) % 2 == 0, "Dataset must contain pairs of human and machine answers, i.e. {Human, Machine, Human...}"
    data_human_tr = data['train']['text'][0::2]
    data_machine_tr = data['train']['text'][1::2]
    data_human_te = data['test']['text'][0::2]
    data_machine_te = data['test']['text'][1::2]
    # Random shuffle the dataset so each pair of answers do not correspond to the same questions
    if SHUFFLE:
        random.shuffle(data_human_tr)
        random.shuffle(data_machine_tr)
        random.shuffle(data_human_te)
        random.shuffle(data_machine_te)
    
    # Allocate data to S1 and S2
    if S1_TYPE == 'human':
        data_s1_tr = data_human_tr
        data_s1_te = data_human_te
    elif S1_TYPE == 'machine':
        data_s1_tr = data_machine_tr
        data_s1_te = data_machine_te
    else:
        raise ValueError("Sample data type not recognized")
    
    if S2_TYPE == 'human':
        data_s2_tr = data_human_tr
        data_s2_te = data_human_te
    elif S2_TYPE == 'machine':
        data_s2_tr = data_machine_tr
        data_s2_te = data_machine_te
    else:
        raise ValueError("Sample data type not recognized")

    # If two sets use the same type, use half of the data of that type for each set so they are disjoint
    if S1_TYPE == S2_TYPE:
        s = len(data_s1_tr)//2
        data_s1_tr = data_s1_tr[:s]
        data_s2_tr = data_s2_tr[s:s*2]
        s = len(data_s1_te)//2
        data_s1_te = data_s1_te[:s]
        data_s2_te = data_s2_te[s:s*2]

    # Set up DK-TST
    dktst = DKTST(
        latent_size_multi=HIDDEN_MULTI,
        device=DEVICE,
        dtype=DTYPE,
        logger=logger
    )
    
    # Load correct checkpoint for continue training
    CONTINUE_EPOCH = 0
    if CONT_CHKPNT:
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
            CONTINUE_EPOCH = chkpnt_epoch_max + 1
            logger.info(f"Continue training from epoch {CONTINUE_EPOCH} for model {CONT_CHKPNT}")
    
    # Basic logging of training parameters
    logger.info(
        f"Training with parameters: \n"
        f"  Checkpoint path={CHKPNT_PATH}\n"
        f"  Continue checkpoint name={CONT_CHKPNT}\n"
        f"  Epoch count={N_EPOCH}\n"
        f"  Hidden size multiplier={HIDDEN_MULTI}\n"
        f"  Dataset={DATASET}\n"
        f"  Dataset LLM={DATASET_LLM}\n"
        f"  S1 type={S1_TYPE}\n"
        f"  S2 type={S2_TYPE}\n"
        f"  Shuffle={SHUFFLE}\n"
        f"  Learning rate={LR}\n"
        f"  Training batch size={BATCH_SIZE_TR}\n"
        f"  Permutatiaon count={PERM_CNT}\n"
        f"  Significance level={SIG_LVL}\n"
        f"  Testing batch size={'In Code' if USE_CUSTOM_TEST else BATCH_SIZE_TE}\n"
        f"  Use custom test={USE_CUSTOM_TEST}\n"
        f"  args.device={DEVICE}\n"
        f"  seed={SEED}\n"
        )
    
    # Start training
    continue_epoch = chkpnt_epoch_max
    J_stars, mmd_values, mmd_stds = dktst.train_and_test(
        s1_tr=data_s1_tr,
        s2_tr=data_s2_tr,
        s1_te=data_s1_te,
        s2_te=data_s2_te,
        lr=LR,
        n_epoch=N_EPOCH,
        batch_size_tr=BATCH_SIZE_TR,
        batch_size_te=BATCH_SIZE_TE,
        save_folder=CHKPNT_PATH,
        perm_cnt=PERM_CNT,
        sig_lvl=SIG_LVL,
        continue_epoch=CONTINUE_EPOCH,
        use_custom_test=USE_CUSTOM_TEST
    )