
from argparse import ArgumentParser
import itertools
import os
from copy import copy

import torch
import pandas as pd
from tqdm import tqdm

import util
from model import DKTST

def log_testing_parameters(args, logger):
    logging_str = f"Testing with parameters: \n"
    for arg, val in args.items():
        logging_str += f"   {arg}={val}\n"
    logger.info(logging_str)
    
def get_test_result(args, logger):
    logger.info("=========== Testing Starts ============\n")

    # Set up dataset
    if not args['single_sample_test']: # Normal two sample test
        logger.info(f'Loading two-sample dataset {args["dataset"]}...')
        _, data_te = util.get_two_sample_datset(
            dataset=args['dataset'],
            dataset_llm=args['dataset_llm'],
            train_ratio=args['dataset_train_ratio'],
            shuffle=args['shuffle'],
            sample_size_test=args['sample_size'],
            sample_size_train=args['sample_size'],
            s1_type=args['s1_type'],
            s2_type=args['s2_type'],
        )
        logger.info(f'Loaded dataset with test size {len(data_te)}...')
    else: # Single sample test
        logger.info(f'Loading single-sample dataset {args["dataset"]}...')
        data_te = util.get_single_sample_datset(
            dataset=args['dataset'],
            dataset_llm=args['dataset_llm'],
            train_ratio=args['dataset_train_ratio'],
            shuffle=args['shuffle'],
            sample_size_test=args['sample_size'],
            test_type=args['single_sample_type'],
            fill_type=args['single_sample_fill_type'],
            test_ratio=args['single_sample_proportion'],
        )
        logger.info(f'Loaded dataset with test size {len(data_te)}...')

    # Set up DK-TST
    logger.info(f'Loading model...')
    model_path = os.path.join(args['model_dir'], args['model_name'])
    hidden_multi = util.Training_Config_Handler.get_train_config(model_path)['hidden_multi']
    dktst = DKTST(
        latent_size_multi=hidden_multi,
        device=args['device'],
        dtype=args['dtype'],
        logger=logger
    )
    
    # Load checkpoint epoch
    best_chkpnt_file_name = None
    chkpnt_file_names = os.listdir(model_path)
    for n in chkpnt_file_names:
        if n.startswith('model_best'):
            best_chkpnt_file_name = n
    if not best_chkpnt_file_name: raise Exception("Did not find the best checkpoint in the model.")
            
    dktst.load(os.path.join(model_path, best_chkpnt_file_name))
    logger.info(f"Using best checkpoint file: {best_chkpnt_file_name}")
    
    log_testing_parameters(args, logger)
    
    # Start testing
    test_power, threshold_avg, mmd_avg = dktst.test(
        data=data_te,
        perm_cnt=args['perm_cnt'],
        sig_lvl=args['sig_lvl'],
        seed=args['seed']
    )
    
    # Logging of test results
    logger.info(
        f"Testing finished with:\n"
        f"{test_power=}\n"
        f"{threshold_avg=}\n"
        f"{mmd_avg=}\n"
    )
    
    return test_power, threshold_avg, mmd_avg, len(data_te)

    
def perform_batch_test(args, logger):
        logger.info("=========== Starting batch test ===========")
        
        # Models' parameters, used to load the trained model.
        # Default assume the use of the latest trained model if multiple ones exist for a configuration
        dataset_tr_list = ['TruthfulQA']
        dataset_llm_tr_list = ['ChatGPT']
        s1s2_tr_list = ['hm']
        shuffle_tr_list = [False] # [True, False]
        linear_size_list = [5] # [3, 5]
        epoch_list = [10000]
        sample_size_tr_list = [2000]
        seed_tr_list = [1103]
        
        # Testing parameters
        dataset_te_list = ['SQuAD1', 'TruthfulQA', 'NarrativeQA']
        dataset_llm_te_list = ['ChatGPT']
        s1s2_te_list = ['hm'] # ['hm', 'hh', 'mm']
        shuffle_te_list = [True] # [True, False]
        sample_size_te_list = [20] # [20, 10, 5, 4, 3]
        sig_lvl_list = [0.05]
        perm_cnt_list = [200] # [20, 50, 100, 200, 400]
        seed_te_list = [1103]
        single_sample_test_list = [True] # 
        single_sample_test_type_list = ['hm', 'mh', 'hh', 'mm'] # ['hm', 'hh', 'mm']
        single_sample_proportion_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # [0.2, 0.5, 0.8]
        
        # Calculate number of tests
        test_count = (
            len(dataset_tr_list) 
            * len(dataset_llm_tr_list)
            * len(s1s2_tr_list)
            * len(shuffle_tr_list)
            * len(linear_size_list)
            * len(epoch_list)
            * len(sample_size_tr_list)
            * len(seed_tr_list)
            * len(dataset_te_list)
            * len(dataset_llm_te_list)
            * len(sig_lvl_list)
            * len(perm_cnt_list)
            * len(shuffle_te_list)
            * len(s1s2_te_list)
            * len(sample_size_te_list)
            * len(seed_te_list)
            * len(single_sample_test_list)
            * len(single_sample_test_type_list)
            * len(single_sample_proportion_list)
        )
        
        # Iterate through model parameters
        results = []
        model_names = os.listdir(args['model_dir'])
        
        with tqdm(total=test_count, desc="Batch Test Progress") as pbar: 
            for (
                data_tr, 
                data_llm_tr, 
                s1s2_tr, 
                shuffle_tr, 
                lin_size, 
                epoch, 
                sample_size_tr, 
                seed_tr
                ) in itertools.product(
                    dataset_tr_list, 
                    dataset_llm_tr_list, 
                    s1s2_tr_list, 
                    shuffle_tr_list, 
                    linear_size_list, 
                    epoch_list, 
                    sample_size_tr_list, 
                    seed_tr_list):
                    
                # Find the lateset version of the checkpoint with this set of parameters
                model_prefix = f"{data_tr}_{s1s2_tr}_{'s' if shuffle_tr else 'nos'}_{lin_size}_{epoch}_{sample_size_tr}_{seed_tr}"
                
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
                    sample_size_te, 
                    seed_te,
                    sst_te,
                    sst_type_te,
                    sst_proportion_te,
                    ) in itertools.product(
                        dataset_te_list, 
                        dataset_llm_te_list, 
                        sig_lvl_list, 
                        perm_cnt_list, 
                        shuffle_te_list, 
                        s1s2_te_list, 
                        sample_size_te_list, 
                        seed_te_list,
                        single_sample_test_list,
                        single_sample_test_type_list,
                        single_sample_proportion_list):
                        
                    str2type = lambda s : 'human' if s == 'h' else 'machine'
                    s1 = str2type(s1s2_te[0])
                    s2 = str2type(s1s2_te[1])
                    sst_test_type = str2type(sst_type_te[0])
                    sst_fill_type = str2type(sst_type_te[1])

                    new_args = copy(args)
                    new_args['model_name'] = model_name
                    new_args['dataset'] = data_te
                    new_args['dataset_llm'] = data_llm_te
                    new_args['dataset_train_ratio'] = 0.8
                    new_args['s1_type'] = s1
                    new_args['s2_type'] = s2
                    new_args['shuffle'] = shuffle_te
                    new_args['perm_cnt'] = perm_cnt
                    new_args['sig_lvl'] = sig_lvl
                    new_args['sample_size'] = sample_size_te
                    new_args['seed'] = seed_te
                    new_args['single_sample_test'] = sst_te
                    new_args['single_sample_type'] = sst_test_type
                    new_args['single_sample_fill_type'] = sst_fill_type
                    new_args['single_sample_proportion'] = sst_proportion_te
                    
                    test_power, threshold_avg, mmd_avg, test_size = get_test_result(args=new_args, logger=logger)
                    
                    results.append({ # Add each result row as a dict into a list of rows
                        'Train - Model Name': model_name,
                        'Train - Linear Layer Size Multiple': lin_size,
                        'Train - Dataset Name': data_tr,
                        'Train - Dataset LLM Name': data_llm_tr,
                        'Train - S1 S2 Type': s1s2_tr,
                        'Train - Shuffled': shuffle_tr,
                        'Train - Epoch Count': epoch,
                        'Train - Sample Size': sample_size_tr,
                        'Test - Dataset Name': data_te,
                        'Test - Dataset LLM Name': data_llm_te,
                        'Test - S1 S2 Type': s1s2_te,
                        'Test - Significance Level': sig_lvl,
                        'Test - Permutation Count': perm_cnt,
                        'Test - Shuffled': shuffle_te,
                        'Test - Sample Size': sample_size_te,
                        'Test - Seed': seed_te,
                        'Test - Test Size': test_size,
                        'Test - Single Sample Test': sst_te,
                        'Test - Single Sample Test Type': sst_test_type,
                        'Test - Single Sample Fill Type': sst_fill_type,
                        'Test - Single Sample Proportion': sst_proportion_te,
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
    
    # Per Run Parameters
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--device', '-dv', type=str, default='auto')
    parser.add_argument('--debug', default=False, action='store_true')
    
    # Batch Test Parameters
    '''If batch test flag is set, all paramters below do not matter as the tested models and paramters are set within code'''
    parser.add_argument('--batch_test', default=False, action='store_true')
    
    # Model Parameters
    parser.add_argument('--model_name', type=str)
    
    # Test Paramters
    parser.add_argument('--dataset', '-d', type=str, default='SQuAD1') # TruthfulQA, SQuAD1, NarrativeQA
    parser.add_argument('--dataset_llm', '-dl', type=str, default='ChatGPT') # ChatGPT, BloomZ, ChatGLM, Dolly, ChatGPT-turbo, GPT4, StableLM
    parser.add_argument('--dataset_train_ratio', type=float, default=0.8) # How much data is allocated to the training set (and testing set)
    parser.add_argument('--s1_type', type=str, default='human') # Type of data (human or machine) for the first sample set
    parser.add_argument('--s2_type', type=str, default='machine') # Type of data (human or machine) for the second sample set
    parser.add_argument('--shuffle', default=False, action='store_true') # Shuffle make sure each pair of s1, s2 samples do not correspond to the same question
    parser.add_argument('--perm_cnt', '-pc', type=int, default=200)
    parser.add_argument('--sig_lvl', '-a', type=float, default=0.05)
    parser.add_argument('--sample_size', '-bte', type=int, default=20)
    parser.add_argument('--seed', '-s', type=int, default=1102) # dimension of samples (default value is 10)
    
    # Single Set Test Parameters
    '''Single Sample Test represent the situation when you only have some sample written by either human or machine, but you
    do not know which. In this case, two sample batches (or sets) are created by putting the data in one of the sample set, 
    and fill the rest of the data with traininig (seen) data of a single type (either human or machine written).''' 
    parser.add_argument('--single_sample_test', default=False, action='store_true')
    parser.add_argument('--single_sample_type', type=str) # Type of data (human or machine)
    parser.add_argument('--single_sample_fill_type', type=str) # Type of data (human or machine)
    parser.add_argument('--single_sample_proportion', type=float) # Proportion of single set data in a test batch
    
    args = parser.parse_args()
    
    # derived parameters
    auto_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = auto_device if args.device == 'auto' else args.device
    # fixed parameters
    args.dtype = torch.float
    return vars(args)

def main():
    start_time_str = util.get_current_time_str()
    args = get_args()
    
    logger = util.setup_logs(
        file_path=f'./test_logs/test_{start_time_str}.log',
        id=start_time_str,
        is_debug=args['debug']
    )
    
    util.setup_seeds(args['seed'])
    
    if args['batch_test']:
        result_df = perform_batch_test(args=args, logger=logger)
        if not args['debug']:
            output_df_to_csv_file(df=result_df, start_time_str=start_time_str, logger=logger)
    else:
        _, _, _, _ = get_test_result(args=args, logger=logger)
        
if __name__ == "__main__":
    main()


    

                    
                    
                    
