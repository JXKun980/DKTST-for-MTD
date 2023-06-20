import logging
from datetime import datetime
from argparse import ArgumentParser
import random
import itertools
from os import listdir

import numpy as np
import torch
import pandas as pd
import re

import dataset_loader as dataset_loader
from DKTST import DKTST

def test(chkpnt_dir, chkpnt_name, dataset_name, dataset_LLM, test_set_only, s1_type, s2_type, 
         shuffle, perm_cnt, sig_lvl, batch_size, seed, device, dtype, logger):
    # Input validation
    assert s1_type in ['human', 'machine'] and s2_type in ['human', 'machine'], "S1 and S2 type must be one of: human, machine."
    
    # Global variable
    chkpnt_path = f'{chkpnt_dir}/{chkpnt_name}'
    
    logger.info("\n============ Testing Starts ============\n")
    
    # Setup seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Set up dataset
    logger.info(f'Loading dataset {dataset_name}...')
    data = dataset_loader.load(name=dataset_name, detectLLM=dataset_LLM)
    assert len(data['train']['text']) % 2 == 0 and len(data['test']['text']) % 2 == 0, "Dataset must contain pairs of human and machine answers, i.e. {Human, Machine, Human...}"
    data_train_human = data['train']['text'][0::2]
    data_train_machine = data['train']['text'][1::2]
    data_test_human = data['test']['text'][0::2]
    data_test_machine = data['test']['text'][1::2]
    
    # Choose to use whole set or only the test set
    if not test_set_only:
        data_test_human = data_train_human + data_test_human
        data_test_machine = data_train_machine + data_test_machine
    
    # Random shuffle the dataset so each pair of answers do not correspond to the same questions
    if shuffle:
        random.shuffle(data_test_human)
        random.shuffle(data_test_machine)
        
    # Allocate data to S1 and S2    
    if s1_type == 'human':
        data_test_s1 = data_test_human
    elif s1_type == 'machine':
        data_test_s1 = data_test_machine
    else:
        raise ValueError("Sample data type not recognized")
    
    if s2_type == 'human':
        data_test_s2 = data_test_human
    elif s2_type == 'machine':
        data_test_s2 = data_test_machine
    else:
        raise ValueError("Sample data type not recognized")

    # If two sets use the same type, use half of the data of that type for each set so they are disjoint
    if s1_type == s2_type:
        s = len(data_test_s1)//2
        data_test_s1 = data_test_s1[:s]
        data_test_s2 = data_test_s2[s:s*2]

    # Set up DK-TST
    hidden_multi = int(chkpnt_name.split('_')[3])
    dktst = DKTST(
        device=device,
        dtype=dtype,
        latent_size_multi=hidden_multi,
        logger=logger
    )
    
    # Load checkpoint epoch
    chkpnt_epoch = int(chkpnt_name.split('_')[4])
    dktst.load(f'{chkpnt_path}/model_ep_{chkpnt_epoch}.pth')
    test_power, threshold_avg, mmd_avg = dktst.test(
        s1=data_test_s1,
        s2=data_test_s2,
        batch_size=batch_size,
        perm_cnt=perm_cnt,
        sig_level=sig_lvl,
        seed=seed
    )
    
    # Basic logging of testing parameters
    logger.info(
        f"Testing with parameters: \n"
        f"  dataset={dataset_name}\n"
        f"  s1 type={s1_type}\n"
        f"  s2 type={s2_type}\n"
        f"  shuffle={shuffle}\n"
        f"  hidden size multiplier={hidden_multi}\n"
        f"  permutatiaon count={perm_cnt}\n"
        f"  significance level={sig_lvl}\n"
        f"  testing batch size={batch_size}\n"
        f"  args.device={device}\n"
        f"  seed={seed}\n"
        f"  checkpoint dir={chkpnt_dir}\n"
        f"  model name={chkpnt_name}\n"
        f"  testing epoch={chkpnt_epoch}")
    
    # Start testing
    logger.info(
        f"Testing finished with:\n"
        f"test_power={test_power}\n"
        f"threshold_avg={threshold_avg}\n"
        f"mmd_avg={mmd_avg}\n"
        )
    
    return test_power, threshold_avg, mmd_avg
    
if __name__ == "__main__":
    # Setup parser
    parser = ArgumentParser()
    # checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_name', type=str)
    parser.add_argument('--checkpoint_epoch', type=int, default=999)
    # test parameters
    parser.add_argument('--dataset', '-d', type=str, default='SQuAD1') # TruthfulQA, SQuAD1, NarrativeQA
    parser.add_argument('--dataset_LLM', '-dl', type=str, default='ChatGPT') # ChatGPT, BloomZ, ChatGLM, Dolly, ChatGPT-turbo, GPT4, StableLM
    parser.add_argument('--test_set_only', default=False, action='store_true') # Whether to use the whole set or only the split test set as the data source
    parser.add_argument('--s1_type', type=str, default='human') # Type of data (human or machine) for the first sample set
    parser.add_argument('--s2_type', type=str, default='machine') # Type of data (human or machine) for the second sample set
    parser.add_argument('--shuffle', default=False, action='store_true') # Shuffle make sure each pair of answers do not correspond to the same questions
    parser.add_argument('--perm_count', '-pc', type=int, default=200)
    parser.add_argument('--sig_level', '-a', type=float, default=0.05)
    parser.add_argument('--batch_size', '-bte', type=int, default=200)
    parser.add_argument('--seed', '-s', type=int, default=1102) # dimension of samples (default value is 10)
    # other parameters
    parser.add_argument('--device', '-dv', type=str, default='auto')
    parser.add_argument('--debug', default=False, action='store_true')
    
    # If custom test flag is set, all parameters except 'test_set_only' and 'other parameters' are ignored, and instead are specified in the code
    parser.add_argument('--custom_test', default=False, action='store_true')
    args = parser.parse_args()
    
    # Setup model parameters
    CHKPNT_DIR = args.checkpoint_dir
    CHKPNT_NAME = args.checkpoint_name
    CHKPNT_EPOCH = args.checkpoint_epoch
    # Setup test parameters
    DATASET = args.dataset
    DATASET_LLM = args.dataset_LLM
    TEST_SET_ONLY = args.test_set_only
    S1_TYPE = args.s1_type
    S2_TYPE = args.s2_type
    SHUFFLE_TEST = args.shuffle
    PERM_CNT = args.perm_count # Amount of permutation during two sample test
    SIG_LEVEL = args.sig_level
    BATCH_SIZE_TEST = args.batch_size
    SEED_TEST = args.seed
    # Set up other parameters
    auto_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = auto_device if args.device == 'auto' else args.device
    DEBUG = args.debug
    DTYPE = torch.float
    
    # Setup logs
    log_level = logging.debug if DEBUG else logging.INFO
    start_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=log_level, 
        force=True,
        handlers=[
            logging.FileHandler(f'./test_logs/testing_{start_time_str}.log', mode='w'),
            logging.StreamHandler()])
    logger = logging.getLogger('test_logger') # Unique per training
    
    if args.custom_test:
        results = []
        
        # Models' parameters
        # Default assume the use of the latest checkpoint if multiple ones exist for a configuration
        dataset_tr_list = ['SQuAD1']
        dataset_llm_tr_list = ['ChatGPT']
        s1s2_tr_list = ['hm']
        shuffle_tr_list = [True] # [True, False]
        linear_size_list = [3] # [3, 5]
        epoch_list = [4000]
        batch_size_tr_list = [2000]
        seed_tr_list = [1102]
        
        # Testing parameters
        dataset_te_list = ['SQuAD1']
        dataset_llm_te_list = ['ChatGPT']
        s1s2_te_list = ['hm', 'hh', 'mm']
        shuffle_te_list = [True, False]
        batch_size_te_list = [20, 10, 5, 4, 3]
        sig_lvl_list = [0.05]
        perm_cnt_list = [200]
        seed_te_list = [1102]
        
        # Iterate through model parameters
        chkpnts = listdir(CHKPNT_DIR)
        for (data_tr, data_llm_tr, s1s2_tr, shuffle_tr, lin_size, epoch, 
             batch_size_tr, seed_tr
             ) in itertools.product(dataset_tr_list, dataset_llm_tr_list, s1s2_tr_list, 
                                    shuffle_tr_list, linear_size_list, epoch_list, 
                                    batch_size_tr_list, seed_tr_list):
            # Find the lateset version of the checkpoint with this set of parameters
            chkpnt_prefix = f"{data_tr}_{s1s2_tr}_{'s' if shuffle_tr else 'nos'}_{lin_size}_{epoch}_{batch_size_tr}_{seed_tr}"
            id_max = 0
            for c in chkpnts:
                if chkpnt_prefix in c:
                    id = int(c.split('_')[-1])
                    if id > id_max:
                        id_max = id
            if id_max == 0:
                logger.error(f"A checkpoint cannot be found for the configuration {chkpnt_prefix}, its test will be skipped.")
                continue
            chkpnt_name = f"{chkpnt_prefix}_{str(id_max)}"
            
            # Iterate through testing parameters
            for (data_te, data_llm_te, sig_lvl, perm_cnt, shuffle_te, s1s2_te, batch_size_te, seed_te) in itertools.product(
                    dataset_te_list, dataset_llm_te_list, sig_lvl_list, perm_cnt_list, shuffle_te_list, 
                    s1s2_te_list, batch_size_te_list, seed_te_list):
                str2type = lambda s : 'human' if s == 'h' else 'machine'
                s1 = str2type(s1s2_te[0])
                s2 = str2type(s1s2_te[1])

                test_power, threshold_avg, mmd_avg = test(CHKPNT_DIR, chkpnt_name, data_te, data_llm_te, 
                                                          TEST_SET_ONLY, s1, s2, shuffle_te, perm_cnt,
                                                          sig_lvl, batch_size_te, seed_te, DEVICE, DTYPE, logger)
                results.append({
                    'Train - Checkpoint Name': chkpnt_name,
                    'Train - Linear Layer Size Multiple': lin_size,
                    'Train - Dataset Name': data_tr,
                    'Train - Dataset LLM Name': data_llm_tr,
                    'Train - S1 S2 Type': s1s2_tr,
                    'Train - Shuffled': shuffle_tr,
                    'Train - Epoch Count': epoch,
                    'Train - Batch Size': batch_size_tr,
                    'Test - Dataset Name': data_te,
                    'Test - Dataset LLM Name': data_llm_te,
                    'Test - S1 S2 Type': s1s2_te,
                    'Test - Significance Level': sig_lvl,
                    'Test - Permutation Count': perm_cnt,
                    'Test - Shuffled': shuffle_te,
                    'Test - Batch Size': batch_size_te,
                    'Test - Seed': seed_te,
                    'Result - Test Power': test_power,
                    'Result - Threshold (Avg.)': threshold_avg,
                    'Result - MMD (Avg.)': mmd_avg})
        
        # Convert to pandas dataframe and output as csv
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'./test_logs/test_{start_time_str}.csv')
        
    else:
        test_power, threshold_avg, mmd_avg = test(CHKPNT_DIR, CHKPNT_NAME, CHKPNT_EPOCH, 
                                                  DATASET, DATASET_LLM, TEST_SET_ONLY, S1_TYPE, S2_TYPE, 
                                                  SHUFFLE_TEST, PERM_CNT, SIG_LEVEL, BATCH_SIZE_TEST, SEED_TEST,
                                                  DEVICE, DTYPE, logger)
                    
                    
                    
