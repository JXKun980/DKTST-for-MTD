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
    parser.add_argument('--dataset', '-d', type=str, default='SQuAD1')
    parser.add_argument('--s1_type', type=str, default='human') # Type of data (human or gpt) for the first sample set
    parser.add_argument('--s2_type', type=str, default='gpt') # Type of data (human or gpt) for the second sample set
    parser.add_argument('--shuffle', default=False, action='store_true') # Shuffle make sure each pair of answers do not correspond to the same questions
    parser.add_argument('--hidden_multi', type=int, default=3) # Hidden dim = In dim * Multiplier
    parser.add_argument('--perm_count', '-pc', type=int, default=200)
    parser.add_argument('--sig_level', '-a', type=float, default=0.05)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.00005)
    parser.add_argument('--batch_size_train', '-btr', type=int, default=2000)
    parser.add_argument('--batch_size_test', '-bte', type=int, default=200) # Not used if custom validation procedure is used
    parser.add_argument('--n_epoch', '-e', type=int, default=1000)
    parser.add_argument('--device', '-dv', type=str, default='auto')
    parser.add_argument('--seed', '-s', type=int, default=1102) # dimension of samples (default value is 10)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--continue_checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    # Setup parameters
    DATASET = args.dataset
    S1_TYPE = args.s1_type
    S2_TYPE = args.s2_type
    SHUFFLE = args.shuffle
    HIDDEN_MULTI = args.hidden_multi
    PERM_CNT = args.perm_count # Amount of permutation during two sample test
    SIG_LEVEL = args.sig_level
    LR = args.learning_rate # default learning rate for MMD-D on HDGM
    BATCH_SIZE_TRAIN = args.batch_size_train
    BATCH_SIZE_TEST = args.batch_size_test
    N_EPOCH = args.n_epoch # number of training epochs
    auto_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = auto_device if args.device == 'auto' else args.device
    SEED = args.seed
    CHECKPOINT_DIR = args.checkpoint_dir
    CONTINUE_CHECKPOINT = args.continue_checkpoint
    DEBUG = args.debug
    DTYPE = torch.float
    
    BATCH_SIZE_TEST = 'Defined in file' # Override as we are using custom test procedure

    # Input validation
    assert S1_TYPE in ['human', 'gpt'] and S1_TYPE in ['human', 'gpt'], "S1 and S2 type must be one of: human, gpt."
        
    # Model directory, unique per run
    if CONTINUE_CHECKPOINT:
        MODEL_DIR = f'{CHECKPOINT_DIR}/{CONTINUE_CHECKPOINT}'
    else:
        datetime_string = datetime.now().strftime("%Y%m%d%H%M%S")
        shuffle_string = 's' if SHUFFLE else 'nos'
        MODEL_DIR = f'{CHECKPOINT_DIR}/{DATASET}_{S1_TYPE[0]}{S2_TYPE[0]}_{shuffle_string}_{HIDDEN_MULTI}_{N_EPOCH}_{BATCH_SIZE_TRAIN}_{SEED}_{datetime_string}'
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
    
    # Setup logs
    log_level = logging.DEBUG if DEBUG else logging.INFO
    logging_path = f'{MODEL_DIR}/training.log'
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=log_level, 
        force=True,
        handlers=[
            logging.FileHandler(f'{MODEL_DIR}/training.log', mode='a'),
            logging.StreamHandler()])
    logger = logging.getLogger(MODEL_DIR) # Unique per training
    logger.info("\n============ Training Starts ============\n")

    # Setup seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Set up dataset
    logger.info(f'Loading dataset {DATASET}...')
    data = dataset_loader.load(DATASET, "")
    assert len(data) % 2 == 0, "Dataset must contain pairs of human and ChatGPT answers, i.e. {Human, ChatGPT, Human...}"
    data_train_human = data['train']['text'][0::2]
    data_train_gpt = data['train']['text'][1::2]
    data_test_human = data['test']['text'][0::2]
    data_test_gpt = data['test']['text'][1::2]
    # Random shuffle the dataset so each pair of answers do not correspond to the same questions
    if SHUFFLE:
        random.shuffle(data_train_human)
        random.shuffle(data_train_gpt)
        random.shuffle(data_test_human)
        random.shuffle(data_test_gpt)
    
    # Allocate data to S1 and S2
    if S1_TYPE == 'human':
        data_train_s1 = data_train_human
        data_test_s1 = data_test_human
    elif S1_TYPE == 'gpt':
        data_train_s1 = data_train_gpt
        data_test_s1 = data_test_gpt
    else:
        raise ValueError("Sample data type not recognized")
    
    if S2_TYPE == 'human':
        data_train_s2 = data_train_human
        data_test_s2 = data_test_human
    elif S2_TYPE == 'gpt':
        data_train_s2 = data_train_gpt
        data_test_s2 = data_test_gpt
    else:
        raise ValueError("Sample data type not recognized")

    # If two sets use the same type, use half of the data of that type for each set so they are disjoint
    if S1_TYPE == S2_TYPE:
        data_train_s1 = data_train_s1[:len(data_train_s1)//2]
        data_train_s2 = data_train_s2[len(data_train_s2)//2:]
        data_test_s1 = data_test_s1[:len(data_test_s1)//2]
        data_test_s2 = data_test_s2[len(data_test_s2)//2:]

    # Set up DK-TST
    dktst = DKTST(
        latent_size_multi=HIDDEN_MULTI,
        device=DEVICE,
        dtype=DTYPE,
        logger=logger
    )
    
    # Load correct checkpoint for continue training
    CONTINUE_EPOCH = 0
    if CONTINUE_CHECKPOINT:
        checkpoint_epoch_max = 0
        checkpoint_file_names = os.listdir(MODEL_DIR)
        for n in checkpoint_file_names:
            match = re.search(r'model_ep_([0-9]+).pth', n)
            if match and int(match.group(1)) > checkpoint_epoch_max:
                checkpoint_epoch_max = int(match.group(1))
        
        if checkpoint_epoch_max == 0:
            raise Exception('Could not find a valid checkpoint to continue')
        else:
            dktst.load(f"{MODEL_DIR}/model_ep_{checkpoint_epoch_max}.pth")
            CONTINUE_EPOCH = checkpoint_epoch_max + 1
            logger.info(f"Continue training from epoch {CONTINUE_EPOCH} for model {CONTINUE_CHECKPOINT}")
    
    # Basic logging of training parameters
    logger.info(
        f"Training with parameters: \n"
        f"  dataset={DATASET}\n"
        f"  s1 type={S1_TYPE}\n"
        f"  s2 type={S2_TYPE}\n"
        f"  shuffle={SHUFFLE}\n"
        f"  hidden size multiplier={HIDDEN_MULTI}\n"
        f"  permutatiaon count={PERM_CNT}\n"
        f"  significance level={SIG_LEVEL}\n"
        f"  learning rate={LR}\n"
        f"  training batch size={BATCH_SIZE_TRAIN}\n"
        f"  testing batch size={BATCH_SIZE_TEST}\n"
        f"  epoch count={N_EPOCH}\n"
        f"  args.device={DEVICE}\n"
        f"  seed={SEED}\n"
        f"  model directory={MODEL_DIR}\n"
        f"  continue checkpoint name={CONTINUE_CHECKPOINT}\n")
    
    # Start training
    continue_epoch = checkpoint_epoch_max
    J_stars, mmd_values, mmd_stds = dktst.train_and_test(
        s1_tr=data_train_s1,
        s2_tr=data_train_s2,
        s1_te=data_test_s1,
        s2_te=data_test_s2,
        lr=LR,
        n_epoch=N_EPOCH,
        batch_size_tr=BATCH_SIZE_TRAIN,
        batch_size_te=BATCH_SIZE_TEST,
        save_folder=MODEL_DIR,
        perm_cnt=PERM_CNT,
        sig_level=SIG_LEVEL,
        continue_epoch=CONTINUE_EPOCH
    )