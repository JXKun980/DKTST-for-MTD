
from argparse import ArgumentParser
import itertools
import os
from copy import copy

import torch
import pandas as pd
from tqdm import tqdm

import util
from model import DKTST

def get_test_result(args, logger):
    # Input validation
    assert args.s1_type in ['human', 'machine'] and args.s2_type in ['human', 'machine'], "S1 and S2 type must be one of: human, machine."
    
    logger.info("=========== Testing Starts ============\n")

    # Set up dataset
    logger.info(f'Loading dataset {args.dataset}...')
    data_human_tr, data_human_te, data_machine_tr, data_machine_te = util.load_data(
        dataset_name=args.dataset,
        llm_name=args.dataset_LLM,
        shuffle=args.shuffle,
        train_ratio=0.8
    )
    
    # Choose to use whole set or only the test set
    if args.use_whole_set:
        data_human_te = data_human_tr + data_human_te
        data_machine_te = data_machine_tr + data_machine_te
    
    # Removed double shuffling bug at 17/08 10:22
        
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
    
    test_set_size = len(data_s1_te)

    # Set up DK-TST
    hidden_multi = int(args.model_name.split('_')[3])
    dktst = DKTST(
        latent_size_multi=hidden_multi,
        device=args.device,
        dtype=args.dtype,
        logger=logger
    )
    
    # Load checkpoint epoch
    best_chkpnt_file_name = None
    chkpnt_file_names = os.listdir(os.path.join(args.model_dir, args.model_name))
    for n in chkpnt_file_names:
        if n.startswith('model_best'):
            best_chkpnt_file_name = n
    if not best_chkpnt_file_name: raise Exception("Did not find the best checkpoint in the model.")
            
    dktst.load(os.path.join(args.model_dir, args.model_name, best_chkpnt_file_name))
    logger.info(f"Using best checkpoint file: {best_chkpnt_file_name}")
    test_power, threshold_avg, mmd_avg = dktst.test(
        s1=data_s1_te,
        s2=data_s2_te,
        batch_size=args.batch_size,
        perm_cnt=args.perm_cnt,
        sig_lvl=args.sig_lvl,
        seed=args.seed
    )
    
    # Basic logging of testing parameters
    logger.info(
        f"Testing with parameters: \n"
        f"    {args.model_dir=}\n"
        f"    {args.model_name=}\n"
        f"    {args.model_epoch=}\n"
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
        f"    {test_set_size=}\n")
    
    # Logging of test results
    logger.info(
        f"Testing finished with:\n"
        f"{test_power=}\n"
        f"{threshold_avg=}\n"
        f"{mmd_avg=}\n"
        )
    
    return test_power, threshold_avg, mmd_avg, test_set_size

    
def perform_batch_test(args, logger):
        logger.info("=========== Starting batch test ===========")
        
        # Models' parameters, used to load the trained model.
        # Default assume the use of the latest trained model if multiple ones exist for a configuration
        dataset_tr_list = ['SQuAD1']
        dataset_llm_tr_list = ['ChatGPT']
        s1s2_tr_list = ['hm']
        shuffle_tr_list = [False] # [True, False]
        linear_size_list = [5] # [3, 5]
        epoch_list = [20000]
        batch_size_tr_list = [2000]
        seed_tr_list = [1103]
        
        # Testing parameters
        dataset_te_list = ['SQuAD1']
        dataset_llm_te_list = ['ChatGPT', 'BloomZ', 'ChatGLM', 'Dolly', 'ChatGPT-turbo', 'GPT4', 'StableLM']
        s1s2_te_list = ['hm', 'hh', 'mm']
        shuffle_te_list = [True] # [True, False]
        batch_size_te_list = [20] # [20, 10, 5, 4, 3]
        sig_lvl_list = [0.05]
        perm_cnt_list = [200] # [20, 50, 100, 200, 400]
        seed_te_list = [1103]
        
        # Calculate number of tests
        test_count = (
            len(dataset_tr_list) 
            * len(dataset_llm_tr_list)
            * len(s1s2_tr_list)
            * len(shuffle_tr_list)
            * len(linear_size_list)
            * len(epoch_list)
            * len(batch_size_tr_list)
            * len(seed_tr_list)
            * len(dataset_te_list)
            * len(dataset_llm_te_list)
            * len(sig_lvl_list)
            * len(perm_cnt_list)
            * len(shuffle_te_list)
            * len(s1s2_te_list)
            * len(batch_size_te_list)
            * len(seed_te_list)
        )
        
        # Iterate through model parameters
        results = []
        model_names = os.listdir(args.model_dir)
        
        with tqdm(total=test_count, desc="Batch Test Progress") as pbar: 
            for (
                data_tr, 
                data_llm_tr, 
                s1s2_tr, 
                shuffle_tr, 
                lin_size, 
                epoch, 
                batch_size_tr, 
                seed_tr
                ) in itertools.product(
                    dataset_tr_list, 
                    dataset_llm_tr_list, 
                    s1s2_tr_list, 
                    shuffle_tr_list, 
                    linear_size_list, 
                    epoch_list, 
                    batch_size_tr_list, 
                    seed_tr_list):
                    
                # Find the lateset version of the checkpoint with this set of parameters
                model_prefix = f"{data_tr}_{s1s2_tr}_{'s' if shuffle_tr else 'nos'}_{lin_size}_{epoch}_{batch_size_tr}_{seed_tr}"
                
                id_max = 0
                for n in model_names:
                    if n.startswith(model_prefix):
                        id = int(n.split('_')[-1])
                        if id > id_max:
                            id_max = id
                if id_max == 0:
                    logger.error(f"A model cannot be found for the configuration {model_prefix}, its test will be skipped.")
                    continue
                
                model_name = f"{model_prefix}_{str(id_max)}"
                
                # Iterate through testing parameters
                for (
                    data_te, 
                    data_llm_te, 
                    sig_lvl, 
                    perm_cnt, 
                    shuffle_te, 
                    s1s2_te, 
                    batch_size_te, 
                    seed_te
                    ) in itertools.product(
                        dataset_te_list, 
                        dataset_llm_te_list, 
                        sig_lvl_list, 
                        perm_cnt_list, 
                        shuffle_te_list, 
                        s1s2_te_list, 
                        batch_size_te_list, 
                        seed_te_list):
                        
                    str2type = lambda s : 'human' if s == 'h' else 'machine'
                    s1 = str2type(s1s2_te[0])
                    s2 = str2type(s1s2_te[1])

                    new_args = copy(args)
                    new_args.model_name = model_name
                    new_args.dataset = data_te
                    new_args.dataset_LLM = data_llm_te
                    new_args.use_whole_set = False # data_te != data_tr
                    new_args.s1_type = s1
                    new_args.s2_type = s2
                    new_args.shuffle = shuffle_te
                    new_args.perm_cnt = perm_cnt
                    new_args.sig_lvl = sig_lvl
                    new_args.batch_size = batch_size_te
                    new_args.seed = seed_te
                    
                    test_power, threshold_avg, mmd_avg, test_size = get_test_result(args=new_args, logger=logger)
                    
                    results.append({ # Add each result row as a dict into a list of rows
                        'Train - Model Name': model_name,
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
                        'Test - Test Size': test_size,
                        'Result - Test Power': test_power,
                        'Result - Threshold (Avg.)': threshold_avg,
                        'Result - MMD (Avg.)': mmd_avg})
                    
                    pbar.update(1) # Update progress bar
        
        # Convert to pandas dataframe and output as csv
        results_df = pd.DataFrame(results)
        return results_df
    
def output_df_to_csv_file(df, start_time_str, logger, dir="./test_logs/"):
    save_file_path = os.path.join(dir, f'test_{start_time_str}.csv')
    df.to_csv(save_file_path)
    logger.info(
        f"Batch test result saved to {save_file_path}\n"
        "=========== Batch test finished ===========\n")
        
def get_args():
    # Setup parser
    parser = ArgumentParser()
    # checkpoint parameters
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_epoch', type=int, default=20000)
    # test parameters
    parser.add_argument('--dataset', '-d', type=str, default='SQuAD1') # TruthfulQA, SQuAD1, NarrativeQA
    parser.add_argument('--dataset_LLM', '-dl', type=str, default='ChatGPT') # ChatGPT, BloomZ, ChatGLM, Dolly, ChatGPT-turbo, GPT4, StableLM
    parser.add_argument('--use_whole_set', default=False, action='store_true') # Whether to use the whole set or only the split test set as the data source
    parser.add_argument('--s1_type', type=str, default='human') # Type of data (human or machine) for the first sample set
    parser.add_argument('--s2_type', type=str, default='machine') # Type of data (human or machine) for the second sample set
    parser.add_argument('--shuffle', default=False, action='store_true') # Shuffle make sure each pair of s1, s2 samples do not correspond to the same question
    parser.add_argument('--perm_cnt', '-pc', type=int, default=200)
    parser.add_argument('--sig_lvl', '-a', type=float, default=0.05)
    parser.add_argument('--batch_size', '-bte', type=int, default=200)
    parser.add_argument('--seed', '-s', type=int, default=1102) # dimension of samples (default value is 10)
    # other parameters
    parser.add_argument('--device', '-dv', type=str, default='auto')
    parser.add_argument('--debug', default=False, action='store_true')
    # If batch test flag is set, all parameters except 'model_dir', 'device' and 'debug' are ignored, and instead are specified in the code
    parser.add_argument('--batch_test', default=False, action='store_true')
    args = parser.parse_args()
    
    # derived parameters
    auto_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = auto_device if args.device == 'auto' else args.device
    # fixed parameters
    args.dtype = torch.float
    return args

def main():
    start_time_str = util.get_current_time_str()
    args = get_args()
    
    logger = util.setup_logs(
        file_path=f'./test_logs/testing_{start_time_str}.log',
        id=start_time_str,
        is_debug=args.debug
    )
    
    util.setup_seeds(args.seed)
    
    if args.batch_test:
        result_df = perform_batch_test(args=args, logger=logger)
        if not args.debug:
            output_df_to_csv_file(df=result_df, start_time_str=start_time_str, logger=logger)
    else:
        _, _, _, _ = get_test_result(args=args, logger=logger)
        
if __name__ == "__main__":
    main()


    

                    
                    
                    
