import itertools
import os
import copy
import re

import torch
import pandas as pd
from tqdm import tqdm
import simple_parsing

import util
from model import DKTST_for_MTD


def log_testing_parameters(args, logger):
    logging_str = f"Testing with parameters: \n"
    for arg, val in args.items():
        logging_str += f"   {arg}={val}\n"
    logger.info(logging_str)
    
def get_test_result(args, logger):
    '''
    Perform a single test with given parameters in args and return the result.
    '''
    logger.info("\n\n=========== Testing Starts ============\n")
    
    util.setup_seeds(args['seed'])

    # Set up dataset
    if args['test_type'] == 'TST': # Normal two sample test
        logger.info(f'Loading two-sample dataset {args["tst_datasets"]}...')
        _, data_te = util.Two_Sample_Dataset.get_train_test_set(
            datasets=args['tst_datasets'], # List of dataset names
            dataset_llm=args['dataset_llm'],
            shuffle=args['shuffle'],
            train_ratio=args['dataset_train_ratio'],
            s1_type=args['tst_s1_type'],
            s2_type=args['tst_s2_type'],
            sample_size_train=args['sample_size'],
            sample_size_test=args['sample_size'],
            device=args['device'],
            sample_count_test=args['sample_count'],
        )
        logger.info(f'Loaded dataset with test size {len(data_te)}...')
        
    elif args['test_type'] == 'SST': # Single sample test
        logger.info(f'Loading single-sample dataset with test set {args["sst_user_dataset"]} and fill set {args["sst_fill_dataset"]}...')
        data_te = util.Single_Sample_Dataset.get_test_set(
            user_dataset = args['sst_user_dataset'], 
            fill_dataset = args['sst_fill_dataset'], 
            dataset_llm = args['dataset_llm'],
            shuffle = args['shuffle'],
            sample_size = args['sample_size'],
            train_ratio = args['dataset_train_ratio'],
            user_type = args['sst_user_type'],
            fill_type = args['sst_fill_type'],
            true_ratio = args['sst_true_ratio'],
            device = args['device'],
            sample_count = args['sample_count'],
        )
        logger.info(f'Loaded dataset with test size {len(data_te)}...')
        
        if args['sst_strong']: # Stronger single sample test (requiring one more dataset)
            logger.info(f'Loading complimentry single-sample dataset with test set {args["sst_user_dataset"]} and fill set {args["sst_fill_dataset"]}...')
            data_te_comp = util.Single_Sample_Dataset.get_test_set(
                user_dataset = args['sst_user_dataset'], 
                fill_dataset = args['sst_fill_dataset'], 
                dataset_llm=args['dataset_llm'],
                shuffle=args['shuffle'],
                sample_size=args['sample_size'],
                train_ratio=args['dataset_train_ratio'],
                user_type=args['sst_user_type'],
                fill_type='human' if args['sst_fill_type'] == 'machine' else 'machine',
                true_ratio=args['sst_true_ratio'],
                device=args['device'],
                sample_count=args['sample_count'],
            )
            logger.info(f'Loaded dataset with test size {len(data_te)}...')


    # Set up DK-TST
    logger.info(f'Loading model...')
    model_path = os.path.join(args['model_dir'], args['model_name'])
    train_config = util.Training_Config_Handler.get_train_config(model_path)
    dktst = DKTST_for_MTD(
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
    if args['test_type'] == 'TST' or (args['test_type'] == 'SST' and not args['sst_strong']): # Normal two sample test and single sample test
        test_power, test_thresholds, test_mmds = dktst.test(
            data=data_te,
            perm_cnt=args['perm_cnt'],
            sig_lvl=args['sig_lvl'],
            seed=args['seed']
        )
        test_threshold_mean = test_thresholds.mean()
        test_mmd_mean = test_mmds.mean()
    elif args['test_type'] == 'SST' and args['sst_strong']: # Strong single-sample test requires additional procedure
        test_power = dktst.strong_single_sample_test(
            data=data_te,
            data_comp=data_te_comp,
            perm_cnt=args['perm_cnt'],
            sig_lvl=args['sig_lvl'],
            seed=args['seed']
        )
        test_threshold_mean = None
        test_mmd_mean = None
    
    # Logging of test results
    logger.info(
        f"Testing finished with:\n"
        f"{test_power=}\n"
        f"{test_threshold_mean=}\n"
        f"{test_mmd_mean=}\n"
    )
    
    return test_power, test_threshold_mean, test_mmd_mean
    
def perform_batch_test_aux(
                       args, # The original arguments are overwritten by the following parameters
                       logger,
                       
                       tr_datasets_list,
                       tr_dataset_llm_list,
                       tr_s1s2_type_list,
                       tr_shuffle_list,
                       tr_linear_size_list,
                       tr_epoch_list,
                       tr_sample_size_list,
                       tr_seed_list,
                       tr_lr_list,
                       tr_chkpnt_ep_list,
                       
                       te_dataset_llm_list,
                       te_shuffle_list,
                       te_sample_size_list,
                       te_sample_cnt_list,
                       te_sig_lvl_list,
                       te_perm_cnt_list,
                       te_seed_list,
                       
                       te_test_type,
                       
                       te_tst_datasets_list,
                       te_tst_s1s2_type_list,
                       
                       te_sst_user_dataset_list,
                       te_sst_fill_dataset_list,
                       te_sst_userfill_type_list,
                       te_sst_true_ratio_list,
                       te_sst_strong_list,
                       ):
        '''
        The function that cross product all the parameters to test for and batch perform the test. Results are saved as CSV file.
        
        @params
            args: The original arguments are overwritten by the following parameters
            logger: The logger to use
            tr_datasets_list: List of datasets the tested model is trained on
            tr_dataset_llm_list: List of LLMs the tested model is trained on
            tr_s1s2_type_list: List of types of data the tested model is trained on
            tr_shuffle_list: List of whether the tested model is trained with shuffled data
            tr_linear_size_list: List of linear layer size of the tested model
            tr_epoch_list: List of epoch count of the tested model
            tr_sample_size_list: List of sample size of the tested model
            tr_seed_list: List of seed of the tested model
            tr_lr_list: List of learning rate of the tested model
            tr_chkpnt_ep_list: List of checkpoint epoch of the tested model
            te_dataset_llm_list: List of LLMs the model is tested on
            te_shuffle_list: List of whether the model is tested with shuffled data
            te_sample_size_list: List of sample size of the model is tested on
            te_sample_cnt_list: List of sample count of the model is tested on
            te_sig_lvl_list: List of significance level of the model is tested on
            te_perm_cnt_list: List of permutation count of the model is tested on
            te_seed_list: List of seed of the model is tested on
            te_test_type: Type of test to run (TST or SST)
            te_tst_datasets_list: List of datasets the model is tested on (TST only)
            te_tst_s1s2_type_list: List of types of data the model is tested on (TST only)
            te_sst_user_dataset_list: List of user datasets the model is tested on (SST only)
            te_sst_fill_dataset_list: List of fill datasets the model is tested on (SST only)
            te_sst_userfill_type_list: List of types of data the model is tested on (SST only)
            te_sst_true_ratio_list: List of true ratio of the model is tested on (SST only)
            te_sst_strong_list: List of whether the model is tested on strong mode (SST only)
        '''
        logger.info("=========== Starting batch test ===========")
        
        # Use default parameters for one of the tests as it is not used
        if te_test_type == 'TST':
            te_sst_user_dataset_list = [None]
            te_sst_fill_dataset_list = [None]
            te_sst_userfill_type_list = [None]
            te_sst_true_ratio_list = [None]
            te_sst_strong_list = [None]
        elif te_test_type == 'SST':
            te_tst_datasets_list = [None]
            te_tst_s1s2_type_list = [None]
        
        # Calculate number of tests
        test_count = (
            len(tr_datasets_list) 
            * len(tr_dataset_llm_list)
            * len(tr_s1s2_type_list)
            * len(tr_shuffle_list)
            * len(tr_linear_size_list)
            * len(tr_epoch_list)
            * len(tr_sample_size_list)
            * len(tr_seed_list)
            * len(tr_lr_list)
            * len(tr_chkpnt_ep_list)
            * len(te_tst_datasets_list)
            * len(te_dataset_llm_list)
            * len(te_sig_lvl_list)
            * len(te_perm_cnt_list)
            * len(te_shuffle_list)
            * len(te_tst_s1s2_type_list)
            * len(te_sample_size_list)
            * len(te_sample_cnt_list)
            * len(te_seed_list)
            * len(te_sst_user_dataset_list)
            * len(te_sst_fill_dataset_list)
            * len(te_sst_userfill_type_list)
            * len(te_sst_true_ratio_list)
            * len(te_sst_strong_list)
        )
        
        # Iterate through model parameters
        results = []
        model_names = os.listdir(args['model_dir'])
        
        with tqdm(total=test_count, desc="Batch Test Progress") as pbar: 
            for (
                tr_datasets, 
                tr_data_llm, 
                tr_s1s2_type, 
                tr_shuffle, 
                tr_lin_size, 
                tr_epoch, 
                tr_sample_size, 
                tr_seed,
                tr_lr,
                tr_chkpnt_ep,
                ) in itertools.product(
                    tr_datasets_list, 
                    tr_dataset_llm_list, 
                    tr_s1s2_type_list, 
                    tr_shuffle_list, 
                    tr_linear_size_list, 
                    tr_epoch_list, 
                    tr_sample_size_list, 
                    tr_seed_list,
                    tr_lr_list,
                    tr_chkpnt_ep_list):
                    
                # Find the lateset version of the model with this set of parameters
                model_prefix = f"{'_'.join(tr_datasets)}_{tr_data_llm}_{tr_s1s2_type}_{'s' if tr_shuffle else 'nos'}_{tr_lin_size}_{tr_epoch}_{tr_sample_size}_{tr_seed}_{tr_lr:.0e}"
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
                    te_data_llm, 
                    te_sig_lvl, 
                    te_perm_cnt, 
                    te_shuffle, 
                    te_sample_size, 
                    te_sample_cnt,
                    te_seed,
                    te_tst_datasets, 
                    te_tst_s1s2_type, 
                    te_sst_user_dataset,
                    te_sst_fill_dataset,
                    te_sst_userfill_type,
                    te_sst_true_ratio,
                    te_sst_strong,
                    ) in itertools.product(
                        te_dataset_llm_list, 
                        te_sig_lvl_list, 
                        te_perm_cnt_list, 
                        te_shuffle_list, 
                        te_sample_size_list, 
                        te_sample_cnt_list,
                        te_seed_list,
                        te_tst_datasets_list, 
                        te_tst_s1s2_type_list, 
                        te_sst_user_dataset_list,
                        te_sst_fill_dataset_list,
                        te_sst_userfill_type_list,
                        te_sst_true_ratio_list,
                        te_sst_strong_list,):
                    
                    new_args = copy.copy(args)
                    new_args['model_name'] = model_name
                    new_args['chkpnt_epoch'] = tr_chkpnt_ep
                    new_args['dataset_llm'] = te_data_llm
                    new_args['dataset_train_ratio'] = 0.8 if tr_datasets == te_tst_datasets else 0.3 # Use more data for testing if the dataset was not used for training
                    new_args['shuffle'] = te_shuffle
                    new_args['perm_cnt'] = te_perm_cnt
                    new_args['sig_lvl'] = te_sig_lvl
                    new_args['sample_size'] = te_sample_size
                    new_args['sample_count'] = te_sample_cnt
                    new_args['seed'] = te_seed
                    
                    new_args['test_type'] = te_test_type
                    
                    str2type = lambda s : 'human' if s == 'h' else 'machine' # Helper function to convert shorthand type to full type names
                    
                    # Derive test types to full name from abbreviation
                    if te_test_type == 'TST':
                        te_tst_s1_type = str2type(te_tst_s1s2_type[0])
                        te_tst_s2_type = str2type(te_tst_s1s2_type[1])
                        te_sst_user_type = None
                        te_sst_fill_type = None
                    elif te_test_type == 'SST':
                        te_sst_user_type = str2type(te_sst_userfill_type[0])
                        te_sst_fill_type = str2type(te_sst_userfill_type[1])
                        te_tst_s1_type = None
                        te_tst_s2_type = None
                        
                    new_args['tst_datasets'] = te_tst_datasets
                    new_args['tst_s1_type'] = te_tst_s1_type
                    new_args['tst_s2_type'] = te_tst_s2_type
                    
                    new_args['sst_user_dataset'] = te_sst_user_dataset
                    new_args['sst_fill_dataset'] = te_sst_fill_dataset
                    new_args['sst_user_type'] = te_sst_user_type
                    new_args['sst_fill_type'] = te_sst_fill_type
                    new_args['sst_true_ratio'] = te_sst_true_ratio
                    new_args['sst_strong'] = te_sst_strong
                    
                    test_power, threshold_mean, mmd_mean = get_test_result(args=new_args, logger=logger)
                    
                    results.append( # Append each result as a new row
                        dict(
                            model_name=model_name,
                            tr_datasets=', '.join(tr_datasets),
                            tr_data_llm=tr_data_llm,
                            tr_s1s2_type=tr_s1s2_type,
                            tr_shuffle=tr_shuffle,
                            tr_lin_size=tr_lin_size,
                            tr_epoch=tr_epoch,
                            tr_sample_size=tr_sample_size,
                            tr_seed=tr_seed,
                            tr_lr=tr_lr,
                            tr_chkpnt_ep=tr_chkpnt_ep,
                            te_data_llm=te_data_llm,
                            te_sig_lvl=te_sig_lvl,
                            te_perm_cnt=te_perm_cnt,
                            te_shuffle=te_shuffle,
                            te_sample_size=te_sample_size,
                            te_sample_cnt=te_sample_cnt,
                            te_seed=te_seed,
                            te_test_type=te_test_type,
                            te_tst_datasets=', '.join(te_tst_datasets),
                            te_tst_s1s2_type=te_tst_s1s2_type,
                            te_sst_user_dataset=te_sst_user_dataset,
                            te_sst_fill_dataset=te_sst_fill_dataset,
                            te_sst_userfill_type=te_sst_userfill_type,
                            te_sst_true_ratio=te_sst_true_ratio,
                            te_sst_strong=te_sst_strong,
                            test_power=test_power,
                            threshold_mean=threshold_mean,
                            mmd_mean=mmd_mean,
                        ))
                    
                    pbar.update(1) # Update progress bar
        
        # Convert to pandas dataframe and output as csv
        results_df = pd.DataFrame(results)
        return results_df
    
    
def perform_batch_test(args, logger):
    '''Perform batch test with the following parameters and save results to file. The script arguments are overwritten by the following parameters.'''
    batch_start_time_str = util.get_current_time_str()
    result_df = perform_batch_test_aux(
        args=args, 
        logger=logger,
        
        # Models' parameters, used to load the trained model.
        # Default assume the use of the latest trained model if multiple ones exist for a configuration
        tr_datasets_list=[['TruthfulQA']], # Note each item is another list of datasets, represnting a merged dataset
        tr_dataset_llm_list=['ChatGPT'],
        tr_s1s2_type_list = ['hm'],
        tr_shuffle_list = [False],
        tr_linear_size_list = [3], # [3, 5]
        tr_epoch_list = [3000],
        tr_sample_size_list = [20],
        tr_seed_list = [1103, 1104, 1105],
        tr_lr_list = [5e-05], # [5e-05, 1e-03]
        tr_chkpnt_ep_list = [None], # [500,1000,2000,3000,4000,5000,6000,7000,8000,9000]
        
        # Shared testing parameters
        te_dataset_llm_list = ['ChatGPT'], # ['ChatGPT', 'ChatGLM', 'Dolly', 'ChatGPT-turbo', 'GPT4', 'StableLM']
        te_shuffle_list = [False], # [True, False]
        te_sample_size_list = [2,3,4,5,6,7,8,9], # [20, 10, 5, 4, 3]
        te_sample_cnt_list = [20],
        te_sig_lvl_list = [0.05],
        te_perm_cnt_list = [200], # [20, 50, 100, 200, 400]
        te_seed_list = [1103],
        
        te_test_type = 'TST', # TST (Two sample test) or SST (Single sample test)
        
        # Two Sample Test only parameters (Will not run if SST is enabled)
        te_tst_datasets_list = [['SQuAD1'], ['TruthfulQA'], ['NarrativeQA']], # Note each item is another list of datasets, represnting a merged dataset
        te_tst_s1s2_type_list = ['hm', 'hh', 'mm'], # Type of data (human or machine) for the first and second sample set   # ['hm', 'hh', 'mm'] 
        
        # Single Sample Test only parameters (If SST is enabled, TST will not run)
        te_sst_user_dataset_list = ['SQuAD1', 'TruthfulQA', 'NarrativeQA'], # ['SQuAD1', 'TruthfulQA', 'NarrativeQA']
        te_sst_fill_dataset_list = ['SQuAD1', 'TruthfulQA', 'NarrativeQA'], # ['SQuAD1', 'TruthfulQA', 'NarrativeQA']
        te_sst_userfill_type_list = ['hm', 'mh', 'hh', 'mm'], # ['hm', 'mh', 'hh', 'mm']
        te_sst_true_ratio_list = [1], # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        te_sst_strong_list = [False],
    )
    if not args['debug']: # Save the result if not in debug mode
        save_file_path = os.path.join('./test_logs', f'test_{batch_start_time_str}.csv')
        result_df.to_csv(save_file_path)
        logger.info(
            f"Batch test result saved to {save_file_path}\n"
            "=========== Batch test finished ===========\n")
    
    
def get_args():
    # Setup parser
    parser = simple_parsing.ArgumentParser()
    
    # Settings Parameters
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory of the trained models.')
    parser.add_argument('--device', '-dv', type=str, default='auto', help='Device to run the model on.')
    parser.add_argument('--debug', default=False, action='store_true', help='Enable debug mode, which supresses file creations.')
    
    # Batch Test Parameters
    parser.add_argument('--batch_test', default=False, action='store_true', help='Whether to run in batch test mode. If yes, the following parameters will be ignored.')
    
    # Model Parameters
    parser.add_argument('--model_name', type=str, help='Name of the model to test.')
    parser.add_argument('--chkpnt_epoch', type=int, help='Epoch count of the checkpoint to load. If not set, the best checkpoint (with file name prefix "model_best_ep_") will be used.')
    
    # Shared Test Paramters
    parser.add_argument('--dataset_llm', '-dl', type=str, default='ChatGPT', help='The LLM the machine generated text is extracted from.') # ChatGPT, BloomZ, ChatGLM, Dolly, ChatGPT-turbo, GPT4, StableLM
    parser.add_argument('--dataset_train_ratio', type=float, default=0.8, help='The proportion of data that is allocated to the training set (remaining is allocated to testing set)')
    parser.add_argument('--shuffle', default=False, action='store_true', help='Enable to shuffle the test set within each distribution to break pair-dependency')
    parser.add_argument('--perm_cnt', '-pc', type=int, default=200, help='Permuatation count for the test')
    parser.add_argument('--sig_lvl', '-a', type=float, default=0.05, help='Significance level for the test')
    parser.add_argument('--sample_size', '-ss', type=int, default=20, help='The amount of samples in each sample set (same for both sample sets in the pair)')
    parser.add_argument('--sample_count', '-sc', type=int, default=50, help='The amount of pairs of sample sets generated for the test')
    parser.add_argument('--seed', '-s', type=int, default=1102, help='Seed of the test')
    parser.add_argument('--test_type', '-tt', type=str, default='TST', help='Type of test to run (TST or SST)')
    
    # Two-sample Test Parameters Only
    parser.add_argument('--tst_datasets', '-d', nargs='+', type=str, help='Dataset(s) to test. If multiple datasets specified, they will be merged into a single dataset.') # TruthfulQA, SQuAD1, NarrativeQA
    parser.add_argument('--tst_s1_type', type=str, default='human', help='Type of data (human or machine) for the first sample set')
    parser.add_argument('--tst_s2_type', type=str, default='machine', help='Type of data (human or machine) for the second sample set')
    
    # Single Set Test Parameters Only
    '''Single Sample Test represent the situation when you only have some sample written by either human or machine, but you
    do not know which. In this case, two sample batches (or sets) are created by putting the data in one of the sample set, 
    and fill the rest of the data with traininig (seen) data of a single type (either human or machine written).''' 
    parser.add_argument('--sst_user_dataset', type=str, help='Dataset used for the user data') # Dataset used for test data
    parser.add_argument('--sst_fill_dataset', type=str, help='Dataset used for the filling data') # Dataset used for fill data
    parser.add_argument('--sst_user_type', type=str, help='Distribution of data for the user dataset (human or machine)')
    parser.add_argument('--sst_fill_type', type=str, help='Distribution of data for the filling dataset (human or machine)')
    parser.add_argument('--sst_true_ratio', type=float, help='Proportion of real data in each sample set that belongs to the user')
    parser.add_argument('--sst_strong', default=False, action='store_true', help='Whether to enable strong mode for the single sample test (requiring two different sets)')
    
    args = parser.parse_args()
    
    # Derived parameters
    auto_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = auto_device if args.device == 'auto' else args.device
    
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
        _, _, _ = get_test_result(args=args, logger=logger)
        
if __name__ == "__main__":
    main()


    

                    
                    
                    
