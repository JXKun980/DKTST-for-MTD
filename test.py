import logging
from datetime import datetime
from argparse import ArgumentParser
import random
import itertools
import os
from copy import copy

import numpy as np
import torch
import pandas as pd
import re

import dataset_loader as dataset_loader
from DKTST import DKTST

def get_test_result(args):
    # Input validation
    assert args.s1_type in ['human', 'machine'] and args.s2_type in ['human', 'machine'], "S1 and S2 type must be one of: human, machine."
    
    LOGGER.info("=========== Testing Starts ============\n")

    # Set up dataset
    LOGGER.info(f'Loading dataset {args.dataset}...')
    data = dataset_loader.load(name=args.dataset, detectLLM=args.dataset_LLM)
    assert len(data['train']['text']) % 2 == 0 and len(data['test']['text']) % 2 == 0, "Dataset must contain pairs of human and machine answers, i.e. {Human, Machine, Human...}"
    data_human_tr = data['train']['text'][0::2]
    data_machine_tr = data['train']['text'][1::2]
    data_human_te = data['test']['text'][0::2]
    data_machine_te = data['test']['text'][1::2]
    
    # Choose to use whole set or only the test set
    if args.use_whole_set:
        data_human_te = data_human_tr + data_human_te
        data_machine_te = data_machine_tr + data_machine_te
    
    # Random shuffle the dataset so each pair of answers do not correspond to the same questions
    if args.shuffle:
        random.shuffle(data_human_te)
        random.shuffle(data_machine_te)
        
    # Allocate data to S1 and S2    
    if args.s1_type == 'human':
        data_s1_te = data_human_te
    elif args.s1_type == 'machine':
        data_s1_te = data_machine_te
    else:
        raise ValueError("Sample data type not recognized")
    
    if args.s2_type == 'human':
        data_s2_te = data_human_te
    elif args.s2_type == 'machine':
        data_s2_te = data_machine_te
    else:
        raise ValueError("Sample data type not recognized")

    # If two sets use the same type, use half of the data of that type for each set so they are disjoint
    if args.s1_type == args.s2_type:
        s = len(data_s1_te)//2
        data_s1_te = data_s1_te[:s]
        data_s2_te = data_s2_te[s:s*2]

    # Set up DK-TST
    hidden_multi = int(args.chkpnt_name.split('_')[3])
    dktst = DKTST(
        latent_size_multi=hidden_multi,
        device=args.device,
        dtype=args.dtype,
        logger=LOGGER
    )
    
    # Load checkpoint epoch
    chkpnt_epoch = int(args.chkpnt_name.split('_')[4])
    dktst.load(f'{args.chkpnt_dir}/{args.chkpnt_name}/model_ep_{chkpnt_epoch}.pth')
    test_power, threshold_avg, mmd_avg = dktst.test(
        s1=data_s1_te,
        s2=data_s2_te,
        batch_size=args.batch_size,
        perm_cnt=args.perm_cnt,
        sig_lvl=args.sig_lvl,
        seed=args.seed
    )
    
    # Basic logging of testing parameters
    LOGGER.info(
        f"Testing with parameters: \n"
        f"    {args.chkpnt_dir=}\n"
        f"    {args.chkpnt_name=}\n"
        f"    {args.chkpnt_epoch=}\n"
        f"    {args.dataset=}\n"
        f"    {args.dataset_LLM=}\n"
        f"    {args.use_whole_set=}\n"
        f"    {args.s1_type=}\n"
        f"    {args.s2_type=}\n"
        f"    {args.shuffle=}\n"
        f"    {args.perm_cnt=}\n"
        f"    {args.sig_lvl=}\n"
        f"    {args.batch_size=}\n"
        f"    {args.seed=}\n"
        f"    {len(data_s1_te)=}\n")
    
    # Logging of test results
    LOGGER.info(
        f"Testing finished with:\n"
        f"{test_power=}\n"
        f"{threshold_avg=}\n"
        f"{mmd_avg=}\n"
        )
    
    return test_power, threshold_avg, mmd_avg

def setup_logs():
    is_debug = ARGS.debug
    log_level = logging.DEBUG if is_debug else logging.INFO
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=log_level,
        force=True,
        handlers=[
            logging.FileHandler(f'./test_logs/testing_{START_TIME_STR}.log', mode='w'),
            logging.StreamHandler()])
    return logging.getLogger('test_logger') # Unique per raining

def setup_seeds():
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed(ARGS.seed)
    torch.backends.cudnn.deterministic = True
    
def perform_batch_test(result_dir="./test_logs/"):
        results = []
        
        # Models' parameters, used to load the trained model.
        # Default assume the use of the latest checkpoint if multiple ones exist for a configuration
        dataset_tr_list = ['SQuAD1']
        dataset_llm_tr_list = ['ChatGPT']
        s1s2_tr_list = ['hm']
        shuffle_tr_list = [False] # [True, False]
        linear_size_list = [5] # [3, 5]
        epoch_list = [15000]
        batch_size_tr_list = [2000]
        seed_tr_list = [1103]
        
        # Testing parameters
        dataset_te_list = ['SQuAD1']
        dataset_llm_te_list = ['ChatGPT']
        s1s2_te_list = ['hm', 'hh', 'mm']
        shuffle_te_list = [True] # [True, False]
        batch_size_te_list = [20] # [20, 10, 5, 4, 3]
        sig_lvl_list = [0.05]
        perm_cnt_list = [20] # [20, 50, 100, 200, 400]
        seed_te_list = [1103]
        
        use_whole_set = False
        
        # Iterate through model parameters
        chkpnts = os.listdir(ARGS.chkpnt_dir)
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
                LOGGER.error(f"A checkpoint cannot be found for the configuration {chkpnt_prefix}, its test will be skipped.")
                continue
            chkpnt_name = f"{chkpnt_prefix}_{str(id_max)}"
            
            # Iterate through testing parameters
            for (data_te, data_llm_te, sig_lvl, perm_cnt, shuffle_te, s1s2_te, batch_size_te, seed_te) in itertools.product(
                    dataset_te_list, dataset_llm_te_list, sig_lvl_list, perm_cnt_list, shuffle_te_list, 
                    s1s2_te_list, batch_size_te_list, seed_te_list):
                str2type = lambda s : 'human' if s == 'h' else 'machine'
                s1 = str2type(s1s2_te[0])
                s2 = str2type(s1s2_te[1])

                new_args = copy(ARGS)
                new_args.chkpnt_name = chkpnt_name
                new_args.dataset = data_te
                new_args.dataset_LLM = data_llm_te
                new_args.use_whole_set = use_whole_set
                new_args.s1_type = s1
                new_args.s2_type = s2
                new_args.shuffle = shuffle_te
                new_args.perm_cnt = perm_cnt
                new_args.sig_lvl = sig_lvl
                new_args.batch_size = batch_size_te
                new_args.seed = seed_te
                
                test_power, threshold_avg, mmd_avg = get_test_result(new_args)
                
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
        results_df.to_csv(os.path.join(result_dir, f'test_{START_TIME_STR}.csv'))
        
def get_args():
    # Setup parser
    parser = ArgumentParser()
    # checkpoint parameters
    parser.add_argument('--chkpnt_dir', type=str, default='./checkpoints')
    parser.add_argument('--chkpnt_name', type=str)
    parser.add_argument('--chkpnt_epoch', type=int, default=999)
    # test parameters
    parser.add_argument('--dataset', '-d', type=str, default='SQuAD1') # TruthfulQA, SQuAD1, NarrativeQA
    parser.add_argument('--dataset_LLM', '-dl', type=str, default='ChatGPT') # ChatGPT, BloomZ, ChatGLM, Dolly, ChatGPT-turbo, GPT4, StableLM
    parser.add_argument('--use_whole_set', default=False, action='store_true') # Whether to use the whole set or only the split test set as the data source
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
    # If batch test flag is set, all parameters except 'device' and 'debug' are ignored, and instead are specified in the code
    parser.add_argument('--batch_test', default=False, action='store_true')
    args = parser.parse_args()
    
    # derived parameters
    auto_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = auto_device if args.device == 'auto' else args.device
    # added parameters
    args.dtype = torch.float
    return args
    
    
if __name__ == "__main__":
    START_TIME_STR = datetime.now().strftime("%Y%m%d%H%M%S")
    ARGS = get_args()
    LOGGER = setup_logs()
    
    setup_seeds()
    
    if ARGS.batch_test:
        perform_batch_test()
    else:
        get_test_result(ARGS)


    

    

                    
                    
                    
