from argparse import ArgumentParser
import os
import re

import torch

from model import DKTST
import util


def get_and_create_model_path(args, start_time_str):
    model_name = (
        f"{args['dataset']}"
        f"_{args['dataset_llm']}"
        f"_{args['s1_type'][0]}{args['s2_type'][0]}"
        f"_{'s' if args['shuffle'] else 'nos'}"
        f"_{args['hidden_multi']}"
        f"_{args['n_epoch']}"
        f"_{args['sample_size_train']}"
        f"_{args['seed']}"
        f"_{args['learning_rate']:.0e}"
        f"_{start_time_str}"
    )
    model_path = (os.path.join(args['model_dir'], model_name))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path


def get_continue_epoch(model_path):
    # Load correct checkpoint for continue training
    chkpnt_epoch_max = 0
    chkpnt_names = os.listdir(model_path)
    for n in chkpnt_names:
        match = re.search(r'model_ep_([0-9]+).pth', n)
        if match and int(match.group(1)) > chkpnt_epoch_max:
            chkpnt_epoch_max = int(match.group(1))
    if chkpnt_epoch_max == 0:
        raise Exception('Could not find a valid checkpoint to continue')
    return chkpnt_epoch_max


def log_training_parameters(args, logger, model_path):
    logging_str = (
        f"Training with parameters: \n"
        f"   {model_path=}\n")
    
    for arg, val in args.items():
        logging_str += f"   {arg}={val}\n"
            
    logger.info(logging_str)


def start_training(args):
    '''Start a training instance based on the arguments'''
    start_time_str = util.get_current_time_str()
    
    util.setup_seeds(args['seed'])
    
    # Get model path
    if args['continue_model']:
        model_path = os.path.join(args['model_dir'], args['continue_model'])
    else:
        model_path = get_and_create_model_path(args, start_time_str)
        
    # Save or load training config
    if args['continue_model']:
        args = util.Training_Config_Handler.load_training_config_to_args(args, model_path) # Override args with loaded args
    else:
        util.Training_Config_Handler.save_training_config(args, model_path)
        
    # Get start epoch
    start_epoch = get_continue_epoch(model_path)+1 if args['continue_model'] else 0
        
    # Set up logger
    logger = util.setup_logs(
        file_path=f'{model_path}/training.log',
        id=start_time_str,
        is_debug=args['debug']
    )
    logger.info("\n\n============ Training Starts ============\n")
    
    # Set up model
    logger.info(f'Setting up model...')
    dktst = DKTST(
        latent_size_multi=args['hidden_multi'],
        device=args['device'],
        dtype=util.str_to_dtype(args['dtype']),
        logger=logger,
        debug=args['debug']
    )
    if args['continue_model']:
        dktst.load(os.path.join(model_path, f'model_ep_{start_epoch-1}.pth'))
        logger.info(f"Continue training from epoch {start_epoch-1} for model {args['continue_model']}")
    
    # Load dataset
    logger.info(f'Loading dataset {args["dataset"]}...')
    
    data_tr, data_te = util.Two_Sample_Dataset.get_train_test_set(
        dataset=args['dataset'],
        dataset_llm=args['dataset_llm'],
        shuffle=args['shuffle'],
        train_ratio=args['dataset_train_ratio'],
        s1_type=args['s1_type'],
        s2_type=args['s2_type'],
        sample_size_train=args['sample_size_train'],
        sample_size_test=args['sample_size_test'],
        device=args['device'],
        sample_count_test=args['sample_count_test'],
    )
    
    logger.info(f'Loaded dataset with training size {len(data_tr)} and test size {len(data_te)}...')
    
    log_training_parameters(args, logger, model_path)
    
    try:
        _, _, _ = dktst.train_and_test(
            data_tr=data_tr,
            data_te=data_te,
            lr=args['learning_rate'],
            total_epoch=args['n_epoch'],
            save_folder=model_path,
            perm_cnt=args['perm_cnt'],
            sig_lvl=args['sig_lvl'],
            start_epoch=start_epoch,
            eval_inteval=args['eval_interval'],
            save_interval=args['save_interval'],
        )
    except util.TrainingError:
        print("Training Error, continue to next run...")
    

def get_args():
    # Setup parser
    parser = ArgumentParser()
    # Per Run Parameters
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--device', '-dv', type=str, default='auto')
    parser.add_argument('--debug', default=False, action='store_true')
    # Continue Parameters
    parser.add_argument('--continue_model', type=str, default=None) # If this is set, all parameters below do not matter
    # Training parameters
    parser.add_argument('--hidden_multi', type=int, default=5) # Hidden dim = In dim * Multiplier
    parser.add_argument('--n_epoch', '-e', type=int, default=8000)
    parser.add_argument('--dataset', '-d', type=str, default='SQuAD1') # TruthfulQA, SQuAD1, NarrativeQA
    parser.add_argument('--dataset_llm', '-dl', type=str, default='ChatGPT') # ChatGPT, BloomZ, ChatGLM, Dolly, ChatGPT-turbo, GPT4, StableLM
    parser.add_argument('--dataset_train_ratio', '-dtrr', type=float, default=0.8)
    parser.add_argument('--s1_type', type=str, default='human') # Type of data (human or machine) for the first sample set
    parser.add_argument('--s2_type', type=str, default='machine') # Type of data (human or machine) for the second sample set
    parser.add_argument('--shuffle', default=False, action='store_true') # Shuffle make sure each pair of answers do not correspond to the same questions
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--sample_size_train', '-sstr', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=100) # Evaluation interval
    parser.add_argument('--save_interval', type=int, default=500) # Evaluation interval
    parser.add_argument('--seed', '-s', type=int, default=1103)
    parser.add_argument('--dtype', type=str, default='float') # Only one of [float, double] is supported. See torch.dtype.
    # Validation parameters
    parser.add_argument('--perm_cnt', '-pc', type=int, default=200)
    parser.add_argument('--sig_lvl', '-a', type=float, default=0.05)
    parser.add_argument('--sample_size_test', '-sste', type=int, default=20) # Not used if custom validation procedure is used
    parser.add_argument('--sample_count_test', '-scte', type=int, default=50)
    args = parser.parse_args()
    
    # Derived parameters
    auto_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = auto_device if args.device == 'auto' else args.device
    
    return vars(args) # return dict

def main():
    args = get_args()


    # Custom run parameter in code
    args['hidden_multi'] = 3
    args['n_epoch'] = 3000
    args['dataset'] = 'NarrativeQA'
    args['dataset_llm'] = 'ChatGPT'
    args['shuffle'] = False
    args['learning_rate'] = 0.00005
    args['sample_size_train'] = 20
    args['seed'] = 1103
    args['sample_count_test'] = 20
    start_training(args)   
    
    # Custom run parameter in code
    args['hidden_multi'] = 3
    args['n_epoch'] = 3000
    args['dataset'] = 'NarrativeQA'
    args['dataset_llm'] = 'ChatGPT'
    args['shuffle'] = False
    args['learning_rate'] = 0.00005
    args['sample_size_train'] = 20
    args['seed'] = 1104
    args['sample_count_test'] = 20
    start_training(args)   
    
    # Custom run parameter in code
    args['hidden_multi'] = 3
    args['n_epoch'] = 3000
    args['dataset'] = 'NarrativeQA'
    args['dataset_llm'] = 'ChatGPT'
    args['shuffle'] = False
    args['learning_rate'] = 0.00005
    args['sample_size_train'] = 20
    args['seed'] = 1105
    args['sample_count_test'] = 20
    start_training(args)   
    
    
    # Continue Training Template
    # args['continue_model'] = 'TruthfulQA_ChatGPT_hm_nos_3_10000_20_1107_2e-04_20230916200738'
    # start_training(args)
    # args['continue_model'] = None
    
    
if __name__ == "__main__":
    main()
