import random
from datetime import datetime
import logging
import os
import copy

import numpy as np
import torch
import yaml

from external import dataset_loader

class Data_Loader:
    def __init__(self, dataset_name, llm_name, s1_type, s2_type, shuffle=False, train_ratio=0.8, use_whole_set_as_test=False) -> None:
        (self.human_tr, 
         self.human_te, 
         self.machine_tr, 
         self.machine_te) = Data_Loader.get_human_machine_data(dataset_name, llm_name, shuffle, train_ratio)
        
        (self.s1_tr, 
         self.s1_te, 
         self.s2_tr, 
         self.s2_te) = self.__get_s1s2_data(s1_type, s2_type)
        
        self.use_whole_set_as_test = use_whole_set_as_test
        if self.use_whole_set_as_test:
            self.s1_te += self.s1_tr
            self.s2_te += self.s2_tr
        
        self.train_size = len(self.s1_tr)
        self.test_size = len(self.s1_te)
        
    def get_human_machine_data(dataset_name, llm_name, shuffle, train_ratio):
        original_data = dataset_loader.load(name=dataset_name, detectLLM=llm_name, train_ratio=train_ratio)
        
        assert len(original_data) % 2 == 0, "Dataset must contain pairs of human and machine answers, i.e. {Human, Machine, Human...}"
        
        human_tr = original_data['train']['text'][0::2]
        human_te = original_data['test']['text'][0::2]
        machine_tr = original_data['train']['text'][1::2]
        machine_te = original_data['test']['text'][1::2]
        
        assert len(human_tr) == len(machine_tr) and len(human_te) == len(machine_te), \
            "Data loaded must have equal number of human and machine data"
        
        # Random shuffle the dataset so each pair of answers do not correspond to the same questions
        if shuffle:
            random.shuffle(human_tr)
            random.shuffle(human_te)
            random.shuffle(machine_tr)
            random.shuffle(machine_te)
        
        return human_tr, human_te, machine_tr, machine_te
        
    def __get_s1s2_data(self, s1_type, s2_type):
        # Input validation
        assert s1_type in ['human', 'machine'] and s2_type in ['human', 'machine'], "S1 and S2 type must be one of: human, machine."
        
        # Allocate data to S1 and S2
        if s1_type == 'human':
            s1_tr = self.data['human']['train']
            s1_te = self.data['human']['test']
        elif s1_type == 'machine':
            s1_tr = self.data['machine']['train']
            s1_te = self.data['machine']['test']
        else:
            raise ValueError("Sample data type not recognized")
        
        if s2_type == 'human':
            s2_tr = self.data['human']['train']
            s2_te = self.data['human']['test']
        elif s2_type == 'machine':
            s2_tr = self.data['machine']['train']
            s2_te = self.data['machine']['test']
        else:
            raise ValueError("Sample data type not recognized")

        # If two sets use the same type, use half of the data of that type for each set so they are disjoint
        if s1_type == s2_type:
            s = len(s1_tr)//2
            s1_tr = s1_tr[:s]
            s2_tr = s2_tr[s:s*2]
            s = len(s1_te)//2
            s1_te = s1_te[:s]
            s2_te = s2_te[s:s*2]
        
        return s1_tr, s1_te, s2_tr, s2_te

class Training_Config_Handler:
    train_config_file_name = 'train_config.yml'
    
    def save_training_config(args, model_path):
        dump_args = { # Only these parameters gets saved with the model
            'n_epoch': args['n_epoch'],
            'hidden_multi': args['hidden_multi'],
            'dataset': args['dataset'],
            'dataset_LLM': args['dataset_LLM'],
            's1_type': args['s1_type'],
            's2_type': args['s2_type'],
            'shuffle': args['shuffle'],
            'learning_rate': args['learning_rate'],
            'batch_size_train': args['batch_size_train'],
            'eval_interval': args['eval_interval'],
            'save_interval': args['save_interval'],
            'seed': args['seed'],
            'perm_cnt': args['perm_cnt'],
            'sig_lvl': args['sig_lvl'],
            'batch_size_test': args['batch_size_test'] if not args['use_custom_test'] else "In Code",
            'use_custom_test': args['use_custom_test'],
        }
        with open(os.path.join(model_path, Training_Config_Handler.train_config_file_name), 'w') as file:
            yaml.dump(dump_args, file)
            
    def get_train_config(model_path):
        with open(os.path.join(model_path, Training_Config_Handler.train_config_file_name), 'r') as file:
            load_args = yaml.safe_load(file)
        return load_args
            
    def load_training_config_to_args(args, model_path):
        train_config = Training_Config_Handler.get_train_config(model_path)
        args_copy = copy.copy(args)
        for k, v in train_config.items():
            args_copy[k] = v
        return args_copy

def setup_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def setup_logs(file_path, id, is_debug=False):
    if is_debug:
        logging.basicConfig(
            format='%(asctime)s %(message)s',
            level=logging.DEBUG,
            force=True,
            handlers=[logging.StreamHandler()])
    else:
        logging.basicConfig(
            format='%(asctime)s %(message)s',
            level=logging.INFO,
            force=True,
            handlers=[
                logging.FileHandler(file_path, mode='a'),
                logging.StreamHandler()])
    return logging.getLogger(id) # Unique per instance
    
def get_current_time_str():
    return datetime.now().strftime("%Y%m%d%H%M%S")