import logging
from argparse import ArgumentParser
import os
import re

import torch

from model import DKTST
import util


def get_and_create_model_path(args, start_time_str):
    if args.continue_model:
        model_path = os.path.join(args.model_dir, args.continue_model)
    else:
        shuffle_str = 's' if args.shuffle else 'nos'
        model_path = (
            f'{args.model_dir}/{args.dataset}_{args.s1_type[0]}{args.s2_type[0]}'
            f'_{shuffle_str}_{args.hidden_multi}_{args.n_epoch}_{args.batch_size_train}_{args.seed}_{start_time_str}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    return model_path

def perform_training(args, model_path, logger):
    # Input validation
    assert args.s1_type in ['human', 'machine'] and args.s2_type in ['human', 'machine'], "S1 and S2 type must be one of: human, machine."
    
    util.setup_seeds(args.seed)
    logger.info("============ Training Starts ============\n")

    # Set up dataset
    logger.info(f'Loading dataset {args.dataset}...')
    data_human_tr, data_human_te, data_machine_tr, data_machine_te = util.load_data(
        dataset_name=args.dataset,
        llm_name=args.dataset_LLM,
        shuffle=args.shuffle,
        train_ratio=0.8
    )
    
    # Allocate data to S1 and S2
    if args.s1_type == 'human':
        data_s1_tr = data_human_tr
        data_s1_te = data_human_te
    elif args.s1_type == 'machine':
        data_s1_tr = data_machine_tr
        data_s1_te = data_machine_te
    else:
        raise ValueError("Sample data type not recognized")
    
    if args.s2_type == 'human':
        data_s2_tr = data_human_tr
        data_s2_te = data_human_te
    elif args.s2_type == 'machine':
        data_s2_tr = data_machine_tr
        data_s2_te = data_machine_te
    else:
        raise ValueError("Sample data type not recognized")

    # If two sets use the same type, use half of the data of that type for each set so they are disjoint
    if args.s1_type == args.s2_type:
        s = len(data_s1_tr)//2
        data_s1_tr = data_s1_tr[:s]
        data_s2_tr = data_s2_tr[s:s*2]
        s = len(data_s1_te)//2
        data_s1_te = data_s1_te[:s]
        data_s2_te = data_s2_te[s:s*2]

    # Set up DK-TST
    dktst = DKTST(
        latent_size_multi=args.hidden_multi,
        device=args.device,
        dtype=args.dtype,
        logger=logger
    )
    
    # Load correct checkpoint for continue training
    cont_epoch = 0
    if args.continue_model:
        chkpnt_epoch_max = 0
        chkpnt_names = os.listdir(model_path)
        for n in chkpnt_names:
            match = re.search(r'model_ep_([0-9]+).pth', n)
            if match and int(match.group(1)) > chkpnt_epoch_max:
                chkpnt_epoch_max = int(match.group(1))
        
        if chkpnt_epoch_max == 0:
            raise Exception('Could not find a valid checkpoint to continue')
        else:
            dktst.load(os.path.join(model_path, f"model_ep_{chkpnt_epoch_max}.pth"))
            cont_epoch = chkpnt_epoch_max + 1
            logger.info(f"Continue training from epoch {cont_epoch} for model {args.continue_model}")
    
    # Basic logging of training parameters
    logging_str = (
        f"Training with parameters: \n"
        f"   {model_path=}\n")
    
    for arg, val in vars(args).items():
        if arg != "batch_size_test":
            logging_str += f"   {arg}={val}\n"
        else:
            logging_str += f"   {arg}={'In Code' if args.use_custom_test else val}\n"
            
    logger.info(logging_str)
    
    # Start training
    J_stars, mmd_values, mmd_stds = dktst.train_and_test(
        s1_tr=data_s1_tr,
        s2_tr=data_s2_tr,
        s1_te=data_s1_te,
        s2_te=data_s2_te,
        lr=args.learning_rate,
        n_epoch=args.n_epoch,
        batch_size_tr=args.batch_size_train,
        batch_size_te=args.batch_size_test,
        save_folder=model_path,
        perm_cnt=args.perm_cnt,
        sig_lvl=args.sig_lvl,
        continue_epoch=cont_epoch,
        use_custom_test=args.use_custom_test,
        eval_inteval=args.eval_interval,
        save_interval=args.save_interval
    )

def get_args():
    # Setup parser
    parser = ArgumentParser()
    # model parameters
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--continue_model', type=str, default=None)
    parser.add_argument('--n_epoch', '-e', type=int, default=8000)
    parser.add_argument('--hidden_multi', type=int, default=5) # Hidden dim = In dim * Multiplier
    # training parameters
    parser.add_argument('--dataset', '-d', type=str, default='SQuAD1') # TruthfulQA, SQuAD1, NarrativeQA
    parser.add_argument('--dataset_LLM', '-dl', type=str, default='ChatGPT') # ChatGPT, BloomZ, ChatGLM, Dolly, ChatGPT-turbo, GPT4, StableLM
    parser.add_argument('--s1_type', type=str, default='human') # Type of data (human or machine) for the first sample set
    parser.add_argument('--s2_type', type=str, default='machine') # Type of data (human or machine) for the second sample set
    parser.add_argument('--shuffle', default=False, action='store_true') # Shuffle make sure each pair of answers do not correspond to the same questions
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--batch_size_train', '-btr', type=int, default=2000)
    parser.add_argument('--eval_interval', type=int, default=100) # Evaluation interval
    parser.add_argument('--save_interval', type=int, default=500) # Evaluation interval
    parser.add_argument('--seed', '-s', type=int, default=1102)
    # validation parameters
    parser.add_argument('--perm_cnt', '-pc', type=int, default=200)
    parser.add_argument('--sig_lvl', '-a', type=float, default=0.05)
    parser.add_argument('--batch_size_test', '-bte', type=int, default=20) # Not used if custom validation procedure is used
    parser.add_argument('--use_custom_test', default=False, action='store_true') # Custom validation that test for a range of batch sizes etc.
    # other parameters
    parser.add_argument('--device', '-dv', type=str, default='auto')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()
    
    # derived parameters
    auto_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = auto_device if args.device == 'auto' else args.device
    # added parameters
    args.dtype = torch.float
    return args

def main():
    start_time_str = util.get_current_time_str()
    args = get_args()
    
    # Custom run parameter in code
    args.n_epoch = 20000
    args.hidden_multi = 5
    args.shuffle = False
    args.dataset = 'TruthfulQA'
    args.seed = 1103
    args.use_custom_test = True
    
    model_path = get_and_create_model_path(args=args, start_time_str=start_time_str)
    
    logger = util.setup_logs(
        file_path=f'{model_path}/training.log',
        id=start_time_str,
        is_debug=args.debug
    )
    
    perform_training(args=args, model_path=model_path, logger=logger)
    
if __name__ == "__main__":
    main()
