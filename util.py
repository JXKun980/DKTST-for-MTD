from __future__ import annotations
from typing import Callable
import random
from datetime import datetime
import logging
import os
import copy
import abc
import math

import numpy as np
import torch
import yaml

from external import dataset_loader

def get_human_machine_dataset(
    dataset: str, 
    dataset_llm: str, 
    shuffle: bool, 
    train_ratio:float,
    ):
    '''Method used to retrieve human and machine data from a database'''
    original_data = dataset_loader.load(name=dataset, detectLLM=dataset_llm, train_ratio=train_ratio)
    
    assert len(original_data) % 2 == 0, "Dataset must contain pairs of human and machine answers, i.e. {Human, Machine, Human...}"
    
    human_tr = original_data['train']['text'][0::2]
    human_te = original_data['test']['text'][0::2]
    machine_tr = original_data['train']['text'][1::2]
    machine_te = original_data['test']['text'][1::2]
    
    assert len(human_tr) == len(machine_tr) and len(human_te) == len(machine_te), "Data loaded must have equal number of human and machine data"
    
    if shuffle: # Individually shuffle each dataset to break the connection between pairs of data
        random.shuffle(human_tr)
        random.shuffle(human_te)
        random.shuffle(machine_tr)
        random.shuffle(machine_te)
        
    return human_tr, human_te, machine_tr, machine_te

def get_two_sample_datset(
    dataset: str, 
    dataset_llm: str,
    shuffle: bool, 
    train_ratio: float,
    s1_type: str,
    s2_type: str,
    sample_size_train: str,
    sample_size_test: str,
    ):

    (human_tr, 
     human_te, 
     machine_tr, 
     machine_te) = get_human_machine_dataset(
         dataset=dataset,
         dataset_llm=dataset_llm,
         shuffle=shuffle,
         train_ratio=train_ratio
     )
     
    data_s1_tr = human_tr if s1_type == 'human' else machine_tr
    data_s2_tr = human_tr if s2_type == 'human' else machine_tr
    data_s1_te = human_te if s1_type == 'human' else machine_te
    data_s2_te = human_te if s2_type == 'human' else machine_te
    if s1_type == s2_type:
        random.shuffle(data_s1_tr)
        random.shuffle(data_s2_tr)
        random.shuffle(data_s1_te)
        random.shuffle(data_s2_te)
     
    data_tr = Two_Sample_Dataset(
        data_s1=data_s1_tr,
        data_s2=data_s2_tr,
        sample_size=sample_size_train,
    )
    data_te = Two_Sample_Dataset(
        data_s1=data_s1_te,
        data_s2=data_s2_te,
        sample_size=sample_size_test,
    )
    return data_tr, data_te


def get_single_sample_datset(
    dataset: str, 
    dataset_llm: str,
    shuffle: bool, 
    train_ratio: float,
    test_type: str,
    fill_type: str,
    test_ratio: float,
    sample_size_test: str,
    ):

    (human_tr, 
     human_te, 
     machine_tr, 
     machine_te) = get_human_machine_dataset(
         dataset=dataset,
         dataset_llm=dataset_llm,
         shuffle=shuffle,
         train_ratio=train_ratio
     )
    data = Single_Sample_Dataset(
        data_test=human_te if test_type == 'human' else machine_te,
        data_fill=human_tr if fill_type == 'human' else machine_tr,
        sample_size=sample_size_test,
        test_ratio=test_ratio
    )
    return data


class Dataset(torch.utils.data.Dataset, abc.ABC):
    '''
    Last sample can be less than the sample size specified
    '''
    def __init__(self, 
                 sample_size: int) -> None:
        self.sample_size = sample_size
    
    @abc.abstractmethod
    def __len__(self):
        pass
    
    @abc.abstractmethod
    def __getitem__(self, idx: int):
        pass
    
    @abc.abstractmethod
    def copy_and_map(self, f: Callable[['list[any]'], 'list[any]']) -> Dataset:
        '''For each data in dataset, data = f(data), return new copy of self'''
        pass
    
    @abc.abstractmethod
    def copy_with_new_sample_size(self, sample_size: int) -> Dataset:
        '''Change sample size, a return a new copy of the Dataset, without copying the concrete data'''
        pass


class Two_Sample_Dataset(Dataset):
    def __init__(self, 
                 sample_size: int, 
                 data_s1: 'list[str]', 
                 data_s2: 'list[str]') -> None:
        
        super().__init__(sample_size)

        self.data_s1 = data_s1
        self.data_s2 = data_s2
    
    def __len__(self):
        return math.ceil(len(self.data_s1) / self.sample_size)
    
    def __getitem__(self, idx):
        lo = idx * self.sample_size
        hi = min(lo + self.sample_size, len(self.data_s1))
        s1 = self.data_s1[lo:hi]
        s2 = self.data_s2[lo:hi]
        return s1, s2
    
    def copy_and_map(self, f):
        return Two_Sample_Dataset(
            sample_size=self.sample_size,
            data_s1=f(self.data_s1),
            data_s2=f(self.data_s2)
        )
        
    def copy_with_new_sample_size(self, sample_size: int):
        return Two_Sample_Dataset(
            sample_size=sample_size,
            data_s1=self.data_s1,
            data_s2=self.data_s2,
        )
        
    def copy_with_single_type(self, use_s1: bool) -> Two_Sample_Dataset:
        '''If dataset contain two types of data, get a copy that contain only one type specified by "use_s1"'''
        return Two_Sample_Dataset(
            sample_size=self.sample_size,
            data_s1=self.data_s1 if use_s1 else self.data_s2,
            data_s2=self.data_s1 if use_s1 else self.data_s2,
        )
    
        
    

class Single_Sample_Dataset(Dataset):
    '''
    The aim of this dataset is only for testing the single-sample capability of trained DKTST-for-MTDs
    
    To get the fill data, the training data set is randomly sampled.
    
    Returns: 
        s1: Contains the test data with some fill data to fill to the "sample size"
            The amount of test data s1 contains is specified by the test_ratio parameter between 0 and 1.
        s2: Contains only fill data of size "sample_size"
    '''
    def __init__(self, 
                 sample_size: int, 
                 data_test: 'list[str]', 
                 data_fill: 'list[str]', 
                 test_ratio: float) -> None:
        # Input validation
        assert 0 < test_ratio < 1, "Test data ratio must be beween 0 and 1 (exclusive)"
        
        super().__init__(sample_size)
        
        self.data_test = data_test
        self.data_fill = data_fill
        self.test_ratio = test_ratio
        self.sample_test_size = round(sample_size * test_ratio)
        
    def __len__(self):
        return math.ceil(len(self.data_test) / self.sample_size)
    
    def __getitem__(self, idx):
        lo = idx * self.sample_test_size
        hi = min(lo + self.sample_test_size, len(self.data_test))
        
        s1_fill_size = self.sample_size - (hi-lo)
        s1_fill_mask = torch.randint(self.data_fill.shape[0], (s1_fill_size,))
        s1 = torch.cat((self.data_test[lo:hi, :], self.data_fill[s1_fill_mask, :]), 0)
        
        s2_fill_mask = torch.randint(self.data_fill.shape[0], (self.sample_size,))
        s2 = self.data_fill[s2_fill_mask, :]
        return s1, s2

    
    def copy_and_map(self, f):
        return Single_Sample_Dataset(
            sample_size=self.sample_size,
            data_test=f(self.data_test),
            data_fill=f(self.data_fill),
            test_ratio=self.test_ratio
        )
        
    def copy_with_new_sample_size(self, sample_size: int):
        return Single_Sample_Dataset(
            sample_size=sample_size,
            data_test=self.data_test,
            data_fill=self.data_fill,
            test_ratio=self.test_ratio
        )  
        

class Training_Config_Handler:
    train_config_file_name = 'train_config.yml'
    
    def save_training_config(args, model_path):
        dump_args = { # Only these parameters gets saved with the model
            'n_epoch': args['n_epoch'],
            'hidden_multi': args['hidden_multi'],
            'dataset': args['dataset'],
            'dataset_llm': args['dataset_llm'],
            's1_type': args['s1_type'],
            's2_type': args['s2_type'],
            'shuffle': args['shuffle'],
            'learning_rate': args['learning_rate'],
            'sample_size_train': args['sample_size_train'],
            'eval_interval': args['eval_interval'],
            'save_interval': args['save_interval'],
            'seed': args['seed'],
            'perm_cnt': args['perm_cnt'],
            'sig_lvl': args['sig_lvl'],
            'sample_size_test': args['sample_size_test'] if not args['use_custom_test'] else "In Code",
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