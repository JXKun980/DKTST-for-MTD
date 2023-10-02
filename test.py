
from argparse import ArgumentParser
import itertools
import os
import copy
import re

import torch
import pandas as pd
from tqdm import tqdm

import util
from model import DKTST

# TODO Separate SST and TST into two interfaces

def log_testing_parameters(args, logger):
    logging_str = f"Testing with parameters: \n"
    for arg, val in args.items():
        logging_str += f"   {arg}={val}\n"
    logger.info(logging_str)
    
def get_test_result(args, logger):
    logger.info("\n\n=========== Testing Starts ============\n")
    
    util.setup_seeds(args['seed'])

    # Set up dataset
    if args['sst_enabled']: # Single sample test
        logger.info(f'Loading single-sample dataset {args["dataset"]}...')
        data_te = util.Single_Sample_Dataset.get_test_set(
            test_dataset = args['sst_test_dataset'], 
            fill_dataset = args['sst_fill_dataset'], 
            dataset_llm = args['dataset_llm'],
            shuffle = args['shuffle'],
            sample_size = args['sample_size'],
            train_ratio = args['dataset_train_ratio'],
            test_type = args['sst_test_type'],
            fill_type = args['sst_fill_type'],
            test_ratio = args['sst_true_ratio'],
            device = args['device'],
            sample_count = args['sample_count'],
        )
        logger.info(f'Loaded dataset with test size {len(data_te)}...')
        
        if args['sst_strong_enabled']: # Stronger single sample test (requiring two different sets)
            logger.info(f'Loading complimentry single-sample dataset {args["dataset"]}...')
            data_te_comp = util.Single_Sample_Dataset.get_test_set(
                test_dataset = args['sst_test_dataset'], 
                fill_dataset = args['sst_fill_dataset'], 
                dataset_llm=args['dataset_llm'],
                shuffle=args['shuffle'],
                sample_size=args['sample_size'],
                train_ratio=args['dataset_train_ratio'],
                test_type=args['sst_test_type'],
                fill_type='human' if args['sst_fill_type'] == 'machine' else 'machine',
                test_ratio=args['sst_true_ratio'],
                device=args['device'],
                sample_count=args['sample_count'],
            )
            logger.info(f'Loaded dataset with test size {len(data_te)}...')
    else: # Normal two sample test
        logger.info(f'Loading two-sample dataset {args["dataset"]}...')
        _, data_te = util.Two_Sample_Dataset.get_train_test_set(
            dataset=args['dataset'],
            dataset_llm=args['dataset_llm'],
            shuffle=args['shuffle'],
            train_ratio=args['dataset_train_ratio'],
            s1_type=args['s1_type'],
            s2_type=args['s2_type'],
            sample_size_train=args['sample_size'],
            sample_size_test=args['sample_size'],
            device=args['device'],
            sample_count_test=args['sample_count'],
        )
        logger.info(f'Loaded dataset with test size {len(data_te)}...')

    # Set up DK-TST
    logger.info(f'Loading model...')
    model_path = os.path.join(args['model_dir'], args['model_name'])
    train_config = util.Training_Config_Handler.get_train_config(model_path)
    dktst = DKTST(
        latent_size_multi=train_config['hidden_multi'],
        device=args['device'],
        dtype=util.str_to_dtype(train_config['dtype']),
        logger=logger
    )
    
    # Load checkpoint epoch
    target_file_name = None
    chkpnt_file_names = os.listdir(model_path)
    for n in chkpnt_file_names:
        if not args['chkpnt_epoch'] and n.startswith('model_best'):
            target_file_name = n
        elif args['chkpnt_epoch']:
            r = re.match(r"model_ep_(\d+).pth", n)
            if r and int(r.group(1)) == args['chkpnt_epoch']:
                target_file_name = n
    if not target_file_name: raise Exception(f"Did not find the checkpoint ({'best' if not args['chkpnt_epoch'] else args['chkpnt_epoch']}) in the model.")
    dktst.load(os.path.join(model_path, target_file_name))
    logger.info(f"Using best checkpoint file: {target_file_name}")
    
    log_testing_parameters(args, logger)
    
    # Start testing
    if args['sst_enabled'] and args['sst_strong_enabled']: # Strong single-sample test requires additional procedure
        test_power = dktst.strong_single_sample_test(
            data=data_te,
            data_comp=data_te_comp,
            perm_cnt=args['perm_cnt'],
            sig_lvl=args['sig_lvl'],
            seed=args['seed']
        )
        test_threshold_mean = None
        test_mmd_mean = None
    else: # Normal two sample test and single sample test
        test_power, test_thresholds, test_mmds = dktst.test(
            data=data_te,
            perm_cnt=args['perm_cnt'],
            sig_lvl=args['sig_lvl'],
            seed=args['seed']
        )
        test_threshold_mean = test_thresholds.mean()
        test_mmd_mean = test_mmds.mean()
    
    # Logging of test results
    logger.info(
        f"Testing finished with:\n"
        f"{test_power=}\n"
        f"{test_threshold_mean=}\n"
        f"{test_mmd_mean=}\n"
    )
    
    return test_power, test_threshold_mean, test_mmd_mean, len(data_te)
    
def perform_batch_test_aux(
                       args, 
                       logger,
                       dataset_tr_list,
                       dataset_llm_tr_list,
                       s1s2_tr_list,
                       shuffle_tr_list,
                       linear_size_list,
                       epoch_list,
                       sample_size_tr_list,
                       seed_tr_list,
                       lr_list,
                       chkpnt_ep_list,
                       dataset_te_list,
                       dataset_llm_te_list,
                       s1s2_te_list,
                       shuffle_te_list,
                       sample_size_te_list,
                       sig_lvl_list,
                       perm_cnt_list,
                       seed_te_list,
                       sst_enabled_list,
                       sst_test_dataset_list,
                       sst_fill_dataset_list,
                       sst_type_list,
                       sst_true_ratio_list,
                       sst_strong_enabled_list,
                       ):
        logger.info("=========== Starting batch test ===========")
        
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
            * len(lr_list)
            * len(chkpnt_ep_list)
            * len(dataset_te_list)
            * len(dataset_llm_te_list)
            * len(sig_lvl_list)
            * len(perm_cnt_list)
            * len(shuffle_te_list)
            * len(s1s2_te_list)
            * len(sample_size_te_list)
            * len(seed_te_list)
            * len(sst_enabled_list)
            * len(sst_test_dataset_list)
            * len(sst_fill_dataset_list)
            * len(sst_type_list)
            * len(sst_true_ratio_list)
            * len(sst_strong_enabled_list)
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
                seed_tr,
                lr,
                chkpnt_ep,
                ) in itertools.product(
                    dataset_tr_list, 
                    dataset_llm_tr_list, 
                    s1s2_tr_list, 
                    shuffle_tr_list, 
                    linear_size_list, 
                    epoch_list, 
                    sample_size_tr_list, 
                    seed_tr_list,
                    lr_list,
                    chkpnt_ep_list):
                    
                # Find the lateset version of the model with this set of parameters
                model_prefix = f"{data_tr}_{data_llm_tr}_{s1s2_tr}_{'s' if shuffle_tr else 'nos'}_{lin_size}_{epoch}_{sample_size_tr}_{seed_tr}_{lr:.0e}"
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
                    sst_enabled_te,
                    sst_test_dataset_te,
                    sst_fill_dataset_te,
                    sst_type_te,
                    sst_true_ratio_te,
                    sst_strong_te,
                    ) in itertools.product(
                        dataset_te_list, 
                        dataset_llm_te_list, 
                        sig_lvl_list, 
                        perm_cnt_list, 
                        shuffle_te_list, 
                        s1s2_te_list, 
                        sample_size_te_list, 
                        seed_te_list,
                        sst_enabled_list,
                        sst_test_dataset_list,
                        sst_fill_dataset_list,
                        sst_type_list,
                        sst_true_ratio_list,
                        sst_strong_enabled_list,):
                        
                    str2type = lambda s : 'human' if s == 'h' else 'machine'
                    s1 = str2type(s1s2_te[0])
                    s2 = str2type(s1s2_te[1])
                    sst_test_type = str2type(sst_type_te[0])
                    sst_fill_type = str2type(sst_type_te[1])

                    new_args = copy.copy(args)
                    new_args['model_name'] = model_name
                    new_args['chkpnt_epoch'] = chkpnt_ep
                    new_args['dataset'] = data_te
                    new_args['dataset_llm'] = data_llm_te
                    new_args['dataset_train_ratio'] = 0.8 if data_tr == data_te else 0.3 # Use more data for testing if the dataset was not used for training
                    new_args['s1_type'] = s1
                    new_args['s2_type'] = s2
                    new_args['shuffle'] = shuffle_te
                    new_args['perm_cnt'] = perm_cnt
                    new_args['sig_lvl'] = sig_lvl
                    new_args['sample_size'] = sample_size_te
                    new_args['sample_count'] = 20
                    new_args['seed'] = seed_te
                    new_args['sst_enabled'] = sst_enabled_te
                    new_args['sst_test_dataset'] = sst_test_dataset_te
                    new_args['sst_fill_dataset'] = sst_fill_dataset_te
                    new_args['sst_test_type'] = sst_test_type
                    new_args['sst_fill_type'] = sst_fill_type
                    new_args['sst_true_ratio'] = sst_true_ratio_te
                    new_args['sst_strong_enabled'] = sst_strong_te
                    
                    test_power, threshold_mean, mmd_mean, test_size = get_test_result(args=new_args, logger=logger)
                    
                    results.append({ # Add each result row as a dict into a list of rows
                        'Train - Model Name': model_name,
                        'Train - Linear Layer Size Multiple': lin_size,
                        'Train - Dataset Name': data_tr,
                        'Train - Dataset LLM Name': data_llm_tr,
                        'Train - S1 S2 Type': s1s2_tr,
                        'Train - Shuffled': shuffle_tr,
                        'Train - Epoch Count': epoch,
                        'Train - Sample Size': sample_size_tr,
                        'Train - Learning Rate': lr,
                        'Train - Checkpoint Epoch': chkpnt_ep,
                        'Train - Seed': seed_tr,
                        'Test - Dataset Name': data_te,
                        'Test - Dataset LLM Name': data_llm_te,
                        'Test - S1 S2 Type': s1s2_te,
                        'Test - Significance Level': sig_lvl,
                        'Test - Permutation Count': perm_cnt,
                        'Test - Shuffled': shuffle_te,
                        'Test - Sample Size': sample_size_te,
                        'Test - Seed': seed_te,
                        'Test - Test Size': test_size,
                        'Test - SST Enabled': sst_enabled_te,
                        'Test - SST Test Dataset': sst_test_dataset_te,
                        'Test - SST Fill Dataset': sst_fill_dataset_te,
                        'Test - SST Test Type': sst_test_type,
                        'Test - SST Fill Type': sst_fill_type,
                        'Test - SST True Data Ratio': sst_true_ratio_te,
                        'Test - SST Strong Enabled': sst_strong_te,
                        'Result - Test Power': test_power,
                        'Result - Threshold Mean': threshold_mean,
                        'Result - MMD Mean': mmd_mean})
                    
                    pbar.update(1) # Update progress bar
        
        # Convert to pandas dataframe and output as csv
        results_df = pd.DataFrame(results)
        return results_df
    
    
def perform_batch_test(args, logger):
      
    batch_start_time_str = util.get_current_time_str()
    result_df = perform_batch_test_aux(
        args=args, 
        logger=logger,
        
        # Models' parameters, used to load the trained model.
        # Default assume the use of the latest trained model if multiple ones exist for a configuration
        dataset_tr_list=['TruthfulQA'],
        dataset_llm_tr_list=['ChatGPT'],
        s1s2_tr_list = ['hm'],
        shuffle_tr_list = [False],
        linear_size_list = [3], # [3, 5]
        epoch_list = [3000],
        sample_size_tr_list = [20],
        seed_tr_list = [1103],
        lr_list = [5e-05], # [5e-05, 1e-03]
        chkpnt_ep_list = [None], # [500,1000,2000,3000,4000,5000,6000,7000,8000,9000]
        
        # Shared testing parameters
        dataset_llm_te_list = ['ChatGPT'], # ['ChatGPT', 'ChatGLM', 'Dolly', 'ChatGPT-turbo', 'GPT4', 'StableLM']
        shuffle_te_list = [False], # [True, False]
        sample_size_te_list = [10], # [20, 10, 5, 4, 3]
        sig_lvl_list = [0.05],
        perm_cnt_list = [200], # [20, 50, 100, 200, 400]
        seed_te_list = [1104],
        
        # Two Sample Test only parameters
        dataset_te_list = ['TruthfulQA'], # ['SQuAD1', 'TruthfulQA', 'NarrativeQA']
        s1s2_te_list = ['hh'], # ['hm', 'hh', 'mm']
        
        # Single Sample Test only parameters (Will not run TST if SST is enabled)
        sst_enabled_list = [True],
        sst_test_dataset_list = ['NarrativeQA'], # ['SQuAD1', 'TruthfulQA', 'NarrativeQA']
        sst_fill_dataset_list = ['NarrativeQA'], # ['SQuAD1', 'TruthfulQA', 'NarrativeQA']
        sst_type_list = ['hm', 'hh', 'mm'], # ['hm', 'hh', 'mm']
        sst_true_ratio_list = [1], # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        sst_strong_enabled_list = [True],
    )
    if not args['debug']: # Save the result if not in debug mode
        save_file_path = os.path.join('./test_logs', f'test_{batch_start_time_str}.csv')
        result_df.to_csv(save_file_path)
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
    parser.add_argument('--model_name', type=str) # Model name you want to load
    parser.add_argument('--chkpnt_epoch', type=int) # Model checkpoint you want to load, if left out, the best checkpoint will be loaded
    
    # Test Paramters
    parser.add_argument('--dataset', '-d', type=str, default='SQuAD1') # TruthfulQA, SQuAD1, NarrativeQA
    parser.add_argument('--dataset_llm', '-dl', type=str, default='ChatGPT') # ChatGPT, BloomZ, ChatGLM, Dolly, ChatGPT-turbo, GPT4, StableLM
    parser.add_argument('--dataset_train_ratio', type=float, default=0.8) # How much data is allocated to the training set (and testing set)
    parser.add_argument('--s1_type', type=str, default='human') # Type of data (human or machine) for the first sample set
    parser.add_argument('--s2_type', type=str, default='machine') # Type of data (human or machine) for the second sample set
    parser.add_argument('--shuffle', default=False, action='store_true') # Shuffle make sure each pair of s1, s2 samples do not correspond to the same question
    parser.add_argument('--perm_cnt', '-pc', type=int, default=200)
    parser.add_argument('--sig_lvl', '-a', type=float, default=0.05)
    parser.add_argument('--sample_size', '-ss', type=int, default=20)
    parser.add_argument('--sample_count', '-sc', type=int, default=50)
    parser.add_argument('--seed', '-s', type=int, default=1102) # dimension of samples (default value is 10)
    
    # Single Set Test Parameters
    '''Single Sample Test represent the situation when you only have some sample written by either human or machine, but you
    do not know which. In this case, two sample batches (or sets) are created by putting the data in one of the sample set, 
    and fill the rest of the data with traininig (seen) data of a single type (either human or machine written).''' 
    parser.add_argument('--sst_enabled', default=False, action='store_true') # Whether to enable single sample test
    parser.add_argument('--sst_test_dataset', type=str) # Dataset used for test data
    parser.add_argument('--sst_fill_dataset', type=str) # Dataset used for fill data
    parser.add_argument('--sst_test_type', type=str) # Type of data (human or machine)
    parser.add_argument('--sst_fill_type', type=str) # Type of data (human or machine)
    parser.add_argument('--sst_true_ratio', type=float) # Ratio of true data in a single sample set
    parser.add_argument('--sst_strong_enabled', default=False, action='store_true') # Whether to enable strong single sample test (requiring two different sets)
    
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
    
    if args['batch_test']:
        perform_batch_test(args=args, logger=logger)
    else:
        _, _, _, _, _ = get_test_result(args=args, logger=logger)
        
if __name__ == "__main__":
    main()


    

                    
                    
                    
