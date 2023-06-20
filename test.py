import logging
from datetime import datetime
from argparse import ArgumentParser
import random

import numpy as np
import torch

import dataset_loader as dataset_loader
from DKTST import DKTST

def test():
    # Input validation
    assert S1_TYPE in ['human', 'gpt'] and S1_TYPE in ['human', 'gpt'], "S1 and S2 type must be one of: human, gpt."
    
    # Global variable
    model_dir = f'{CHECKPOINT_DIR}/{MODEL_NAME}'
    
    # Setup logs
    log_level = logging.DEBUG if DEBUG else logging.INFO
    datetime_string = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=log_level, 
        force=True,
        handlers=[
            logging.FileHandler(f'{model_dir}/testing_{datetime_string}.log', mode='w'),
            logging.StreamHandler()])
    logger = logging.getLogger(datetime_string) # Unique per training
    logger.info("\n============ Testing Starts ============\n")
    
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
    
    # Choose to use whole set or only the test set
    if USE_WHOLE_SET:
        data_test_human = data_train_human + data_test_human
        data_test_gpt = data_train_gpt + data_test_gpt
    
    # Random shuffle the dataset so each pair of answers do not correspond to the same questions
    if SHUFFLE:
        random.shuffle(data_test_human)
        random.shuffle(data_test_gpt)
        
    # Allocate data to S1 and S2    
    if S1_TYPE == 'human':
        data_test_s1 = data_test_human
    elif S1_TYPE == 'gpt':
        data_test_s1 = data_test_gpt
    else:
        raise ValueError("Sample data type not recognized")
    
    if S2_TYPE == 'human':
        data_test_s2 = data_test_human
    elif S2_TYPE == 'gpt':
        data_test_s2 = data_test_gpt
    else:
        raise ValueError("Sample data type not recognized")

    # If two sets use the same type, use half of the data of that type for each set so they are disjoint
    if S1_TYPE == S2_TYPE:
        data_test_s1 = data_test_s1[:len(data_test_s1)//2]
        data_test_s2 = data_test_s2[len(data_test_s2)//2:]

    # Set up DK-TST
    dktst = DKTST(
        device=DEVICE,
        dtype=DTYPE,
        latent_size_multi=HIDDEN_MULTI,
        logger=logger
    )
    
    # Load checkpoint epoch
    dktst.load(f'{model_dir}/model_ep_{CHKPNT_EPOCH}.pth')
    test_power, threshold_avg, mmd_avg = dktst.test(
        s1=data_test_s1,
        s2=data_test_s2,
        batch_size=BATCH_SIZE_TEST,
        perm_cnt=PERM_CNT,
        sig_level=SIG_LEVEL,
        seed=SEED
    )
    
    # Basic logging of testing parameters
    logger.info(
        f"Testing with parameters: \n"
        f"  dataset={DATASET}\n"
        f"  s1 type={S1_TYPE}\n"
        f"  s2 type={S2_TYPE}\n"
        f"  shuffle={SHUFFLE}\n"
        f"  hidden size multiplier={HIDDEN_MULTI}\n"
        f"  permutatiaon count={PERM_CNT}\n"
        f"  significance level={SIG_LEVEL}\n"
        f"  testing batch size={BATCH_SIZE_TEST}\n"
        f"  args.device={DEVICE}\n"
        f"  seed={SEED}\n"
        f"  checkpoint dir={CHECKPOINT_DIR}\n"
        f"  model name={MODEL_NAME}\n"
        f"  testing epoch={CHKPNT_EPOCH}")
    
    # Start testing
    logger.info(
        "Testing finished with:\n"
        f"test_power={test_power}\n"
        f"threshold_avg={threshold_avg}\n"
        f"mmd_avg={mmd_avg}\n"
        )
    
if __name__ == "__main__":
    # Setup parser
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='SQuAD1')
    parser.add_argument('--use_whole_set', default=False, action='store_true') # Whether to use the whole set or only the split test set as the data source
    parser.add_argument('--s1_type', type=str, default='human') # Type of data (human or gpt) for the first sample set
    parser.add_argument('--s2_type', type=str, default='gpt') # Type of data (human or gpt) for the second sample set
    parser.add_argument('--shuffle', default=False, action='store_true') # Shuffle make sure each pair of answers do not correspond to the same questions
    parser.add_argument('--hidden_multi', type=int, default=3) # Hidden dim = In dim * Multiplier
    parser.add_argument('--perm_count', '-pc', type=int, default=200)
    parser.add_argument('--sig_level', '-a', type=float, default=0.05)
    parser.add_argument('--batch_size_test', '-bte', type=int, default=200)
    parser.add_argument('--device', '-dv', type=str, default='auto')
    parser.add_argument('--seed', '-s', type=int, default=1102) # dimension of samples (default value is 10)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--checkpoint_epoch', type=int, default=999)
    parser.add_argument('--custom_test', default=False, action='store_true')
    args = parser.parse_args()
    
    # Setup parameters
    DATASET = args.dataset
    USE_WHOLE_SET = args.use_whole_set
    S1_TYPE = args.s1_type
    S2_TYPE = args.s2_type
    SHUFFLE = args.shuffle
    HIDDEN_MULTI = args.hidden_multi
    PERM_CNT = args.perm_count # Amount of permutation during two sample test
    SIG_LEVEL = args.sig_level
    BATCH_SIZE_TEST = args.batch_size_test
    auto_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = auto_device if args.device == 'auto' else args.device
    SEED = args.seed
    DEBUG = args.debug
    DTYPE = torch.float
    CHECKPOINT_DIR = args.checkpoint_dir
    MODEL_NAME = args.model_name
    CHKPNT_EPOCH = args.checkpoint_epoch
    
    if args.custom_test:
        for (MODEL_NAME, HIDDEN_MULTI) in [
            ('SQuAD1_hg_s_3_4000_2000_1102_20230531184222', 3),
            ('SQuAD1_hg_nos_5_4000_2000_1102_20230601085208', 5),
            ('SQuAD1_hg_s_5_4000_2000_1102_20230601072733', 5)]:
            for SHUFFLE in [True, False]:
                for (S1_TYPE, S2_TYPE) in [('human', 'gpt'), ('human', 'human'), ('gpt', 'gpt')]:
                    for BATCH_SIZE_TEST in [20, 10, 5, 4, 3]:
                        test()
    else:
        test()
                    
                    
                    
