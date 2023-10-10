from __future__ import annotations
from typing import Callable
import random
from datetime import datetime
import logging
import os
import copy
import abc

import numpy as np
import torch
import yaml

from external import dataset_loader

class TrainingError(Exception):
    def __init__(self, message='No Message'):
        self.message = message
        super().__init__(self.message)
        
        
def str_to_dtype(s: str):
    if s == "float":
        return torch.float32
    elif s == "double":
        return torch.double
    else:
        raise ValueError("Unknown data type string.")


class Dataset(abc.ABC):
    '''
    Dataset abstract class that should be extended by a concrete dataset class.
    '''
    def __init__(self, 
                 device: str,
                 sample_size: int,
                 dataset_size: int,
                 sample_count: int=None) -> None:
        '''
        @params
            device: Device to store the data on
            sample_size: Size of each sample set
            dataset_size: Number of samples in the dataset
            sample_count: Number of sample sets to draw from the dataset
        '''
        self.device = device
        self.dataset_size = dataset_size
        
        # Number of data pairs in a sample
        self.set_sample_size(sample_size)
        
        # Number of samples in the dataset.
        self.set_sample_count(sample_count)
        
    def __len__(self):
        return self.sample_count
    
    @abc.abstractmethod
    def __iter__(self):
        pass
    
    def set_sample_size(self, sample_size: int):
        self.__sample_size = sample_size if sample_size <= self.dataset_size else self.dataset_size 
        
    def get_sample_size(self):
        return self.__sample_size
    
    sample_size = property(get_sample_size, set_sample_size) # Registering getter and setter as property
    
    def set_sample_count(self, sample_count: int):
        # If None, default to sample without replacement, so sample count = dataset_size / sample_size.
        # Else, the sample without replacement procedure is performed multiple times until the desired count is returned.
        self.__sample_count = sample_count if sample_count else self.dataset_size // self.sample_size
        
    def get_sample_count(self):
        return self.__sample_count
    
    sample_count = property(get_sample_count, set_sample_count) # Registering getter and setter as property
    
    @abc.abstractmethod
    def copy_and_map(self, f: Callable[['list[any]'], 'list[any]']): # -> Self
        '''For each data list x, x = f(x), return new copy of self containing the new data lists'''
        pass

    def get_human_machine_datalist(
        dataset: str, 
        dataset_llm: str, 
        shuffle: bool, 
        train_ratio:float,
        ):
        '''
        Method used to retrieve human and machine data from a database
        
        @params
            dataset: Name of the dataset to load
            dataset_llm: Name of the LLM to load as the machine set
            shuffle: Whether to shuffle (dependency break) the data
            train_ratio: Ratio of training data to test data
        @returns
            human_tr: List of human training data
            human_te: List of human test data
            machine_tr: List of machine training data
            machine_te: List of machine test data
        '''
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


class Two_Sample_Dataset(Dataset):
    def __init__(self, 
                 data_s1: 'list[str]', 
                 data_s2: 'list[str]',
                 device,
                 sample_size: int,
                 sample_count: int = None
                 ) -> None:
        '''
        @params
            data_s1: List of data from distribution 1
            data_s2: List of data from distribution 2
            device: Device to store the data on
            sample_size: Size of each sample set
            sample_count: Number of sample sets to draw from the dataset
        '''
        super().__init__(
            device = device, 
            sample_size = sample_size,
            dataset_size = len(data_s1),
            sample_count = sample_count
            )
        
        self.data_s1 = np.array(data_s1)
        self.data_s2 = np.array(data_s2)
    
    def __iter__(self):
        def data_generator():
            '''Generator closure function that act as iterator that produce samples by randomly sampling data'''
            sample_cnt = copy.copy(self.sample_count)
            sample_size = copy.copy(self.sample_size)
            device = copy.copy(self.device)
            
            data_s1_tensor = torch.from_numpy(self.data_s1).to(device)
            data_s2_tensor = torch.from_numpy(self.data_s2).to(device)
            
            data_size = data_s1_tensor.shape[0]
            unique_sample_cnt = data_size // sample_size
    
            # Random permute data
            rand_index = torch.randperm(data_size, device=device)
            data_s1_tensor = data_s1_tensor[rand_index]
            data_s2_tensor = data_s2_tensor[rand_index]
            
            i = 0 # Index used to track which portion of the dataset is used as the current sample
            for _ in range(sample_cnt):
                if i >= unique_sample_cnt: # run out of unique data, reshuffle and restart random sampling
                    i = 0
                    # Random permute data
                    rand_index = torch.randperm(data_size, device=device)
                    data_s1_tensor = data_s1_tensor[rand_index]
                    data_s2_tensor = data_s2_tensor[rand_index]
                # Get the next sample_size amount of samples, since they are randomly permuted, it is equivalent to random sampling
                yield data_s1_tensor[i*sample_size : (i+1)*sample_size], data_s2_tensor[i*sample_size : (i+1)*sample_size]
                i += 1

        return data_generator()
    
    def copy_and_map(self, f):
        new_self = copy.copy(self)
        
        data_s1 = copy.deepcopy(self.data_s1)
        data_s2 = copy.deepcopy(self.data_s2)
        
        split = 5
        split_size = len(data_s1) // split
        new_self.data_s1 = f(data_s1[0:split_size])
        new_self.data_s2 = f(data_s2[0:split_size])
        for i in range(1, split):
            new_self.data_s1 = np.concatenate((new_self.data_s1, f(data_s1[i*split_size:(i+1)*split_size])), axis=0)
            new_self.data_s2 = np.concatenate((new_self.data_s2, f(data_s2[i*split_size:(i+1)*split_size])), axis=0)

        return new_self
        
    def copy_with_single_type(self, use_s1: bool) -> Two_Sample_Dataset:
        '''If dataset contain two types of data, get a copy that contain only one type specified by "use_s1"'''
        new_self = copy.copy(self)
        if use_s1:
            new_self.data_s2 = copy.deepcopy(self.data_s1)
            random.shuffle(new_self.data_s2)
        else:
            new_self.data_s1 = copy.deepcopy(self.data_s2)
            random.shuffle(new_self.data_s1)
        return new_self
    
    def get_train_test_set(
            datasets: 'list[str]', 
            dataset_llm: str,
            shuffle: bool, 
            train_ratio: float,
            s1_type: str,
            s2_type: str,
            sample_size_train: int,
            sample_size_test: int,
            device: str,
            sample_count_test: int=None,
            ):
        '''
        Constructor function that construct train and test sets from arguments
        
        @params
            datasets: List of dataset names to load
            dataset_llm: Name of the LLM to load as the machine set
            shuffle: Whether to shuffle (dependency break) the data
            train_ratio: Ratio of training data to test data
            s1_type: Type of data to load for dataset 1
            s2_type: Type of data to load for dataset 2
            sample_size_train: Size of each sample set for training
            sample_size_test: Size of each sample set for testing
            device: Device to store the data on
            sample_count_test: Number of sample sets to draw from the test dataset
        '''

        data_s1_tr_aggregated = np.array([])
        data_s2_tr_aggregated = np.array([])
        data_s1_te_aggregated = np.array([])
        data_s2_te_aggregated = np.array([])
        
        for dataset in datasets:
            (human_tr, 
            human_te, 
            machine_tr, 
            machine_te) = Two_Sample_Dataset.get_human_machine_datalist(
                dataset=dataset,
                dataset_llm=dataset_llm,
                shuffle=shuffle,
                train_ratio=train_ratio
            )
            
            data_s1_tr = copy.deepcopy(human_tr) if s1_type == 'human' else copy.deepcopy(machine_tr)
            data_s2_tr = copy.deepcopy(human_tr) if s2_type == 'human' else copy.deepcopy(machine_tr)
            data_s1_te = copy.deepcopy(human_te) if s1_type == 'human' else copy.deepcopy(machine_te)
            data_s2_te = copy.deepcopy(human_te) if s2_type == 'human' else copy.deepcopy(machine_te)
            if s1_type == s2_type:
                random.shuffle(data_s1_tr)
                random.shuffle(data_s2_tr)
                random.shuffle(data_s1_te)
                random.shuffle(data_s2_te)

            # Concatenate data from different datasets, select only a portion of the data if multiple datasets are used
            # Even though data from different datasets are concatenated, the data will be shuffled during training
            data_s1_tr_aggregated = np.concatenate((data_s1_tr_aggregated, data_s1_tr))
            data_s2_tr_aggregated = np.concatenate((data_s2_tr_aggregated, data_s2_tr))
            data_s1_te_aggregated = np.concatenate((data_s1_te_aggregated, data_s1_te))
            data_s2_te_aggregated = np.concatenate((data_s2_te_aggregated, data_s2_te))
        
        data_tr = Two_Sample_Dataset(
            data_s1=data_s1_tr_aggregated,
            data_s2=data_s2_tr_aggregated,
            device=device,
            sample_size=sample_size_train,
            sample_count=None
        )
        data_te = Two_Sample_Dataset(
            data_s1=data_s1_te_aggregated,
            data_s2=data_s2_te_aggregated,
            device=device,
            sample_size=sample_size_test,
            sample_count=sample_count_test
        )
        return data_tr, data_te

class Single_Sample_Dataset(Dataset):
    '''
    The aim of this dataset is only for testing the single-sample capability of trained DKTST-for-MTDs
    
    To get the fill data, the training data set is randomly sampled.
    '''
    def __init__(self, 
                 data_user: 'list[str]', 
                 data_fill: 'list[str]', 
                 true_ratio: float,
                 device: str,
                 sample_size: int,
                 sample_count: int = None
                 ) -> None:
        '''
        @params
            data_user: List of data from the user data distribution
            data_fill: List of data from the fill data distribution
            true_ratio: Ratio of true user data in the user sample set
            device: Device to store the data on
            sample_size: Size of each sample set
            sample_count: Number of sample sets to draw from the dataset
        '''
        # Input validation
        assert 0 < true_ratio <= 1, "Test data ratio must be in range (0, 1]"
        
        super().__init__(
            device = device, 
            sample_size = sample_size,
            dataset_size = len(data_user),
            sample_count = sample_count
            )
        
        self.data_user = np.array(data_user)
        self.data_fill = np.array(data_fill)
        self.true_ratio = true_ratio
        self.sample_true_size = round(sample_size * true_ratio)
    
    def __iter__(self):
        def data_generator():
            '''Generator closure function that act as iterator that produce samples by randomly sampling data'''
            device = copy.copy(self.device)
            
            data_user_tensor = torch.from_numpy(self.data_user).to(device)
            data_fill_tensor = torch.from_numpy(self.data_fill).to(device)

            user_size = data_user_tensor.shape[0]
            fill_size = data_fill_tensor.shape[0]
            unique_user_sample_cnt = user_size // self.sample_true_size
            
            # Random permute test data
            rand_index = torch.randperm(user_size, device=device)
            data_user_tensor = data_user_tensor[rand_index]
            
            i = 0
            for _ in range(self.sample_count):
                if i >= unique_user_sample_cnt: # run out of unique data, reshuffle and restart random sampling
                    i = 0
                    # Random permute test data
                    rand_index = torch.randperm(user_size, device=device)
                    data_user_tensor = data_user_tensor[rand_index]
                # Construct s1 and s2 from random shuffled test data and train data
                s1_fill_size = self.sample_size - self.sample_true_size
                s1s2_fill_mask = torch.randperm(fill_size, device=device)
                s1_fill_mask = s1s2_fill_mask[:s1_fill_size]
                s2_fill_mask = s1s2_fill_mask[s1_fill_size:s1_fill_size+self.sample_size]
                
                s1_tensor = torch.cat((data_user_tensor[i*self.sample_true_size : (i+1)*self.sample_true_size], data_fill_tensor[s1_fill_mask]), 0)
                s2_tensor = data_fill_tensor[s2_fill_mask]
                
                yield (s1_tensor, s2_tensor)
                i += 1
                
        return data_generator()
    
    def copy_and_map(self, f):
        new_self = copy.copy(self)
        new_self.data_user = f(copy.deepcopy(self.data_user))
        new_self.data_fill = f(copy.deepcopy(self.data_fill))
        return new_self
    
    def get_test_set(
        user_dataset: str, 
        fill_dataset: str,
        dataset_llm: str,
        shuffle: bool,
        train_ratio: float,
        user_type: str,
        fill_type: str,
        true_ratio: float,
        sample_size: str,
        device: str,
        sample_count: int = None,
        ):
        '''
        @params
            user_dataset: Name of the dataset to load as the user data
            fill_dataset: Name of the dataset to load as the fill data
            dataset_llm: Name of the LLM to load as the machine set
            shuffle: Whether to shuffle (dependency break) the data
            train_ratio: Ratio of training data to test data
            user_type: Type of data to load for user data
            fill_type: Type of data to load for fill data
            true_ratio: Ratio of true user data in the user sample set
            sample_size: Size of each sample set
            device: Device to store the data on
            sample_count: Number of sample sets to draw from the dataset
        '''

        _, human_test, _, machine_test = Single_Sample_Dataset.get_human_machine_datalist(
            dataset=user_dataset,
            dataset_llm=dataset_llm,
            shuffle=shuffle,
            train_ratio=train_ratio
        ) # Only get test split for the test data
        
        human_fill, _, machine_fill, _ = Single_Sample_Dataset.get_human_machine_datalist(
            dataset=fill_dataset,
            dataset_llm=dataset_llm,
            shuffle=shuffle,
            train_ratio=train_ratio
        ) # Only get the training split for the fill data
        
        data = Single_Sample_Dataset(
            data_user = copy.deepcopy(human_test) if user_type == 'human' else copy.deepcopy(machine_test),
            data_fill = copy.deepcopy(human_fill) if fill_type == 'human' else copy.deepcopy(machine_fill),
            true_ratio=true_ratio,
            device=device,
            sample_size=sample_size,
            sample_count=sample_count,
        )
        return data


class Training_Config_Handler:
    '''Loading and saving training config in YAML format'''
    train_config_file_name = 'train_config.yml'
    
    def save_training_config(args: dict, model_path):
        dump_args = copy.copy(args)
        exclude_args = ['model_dir', 'device', 'debug', 'continue_model']
        for ea in exclude_args:
            del dump_args[ea]
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    # Using export CUBLAS_WORKSPACE_CONFIG=:4096:8 stuck for 10 minutes whtiuot proceeding
    
def setup_logs(file_path, id, is_debug=False, supress_file=False):
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
            handlers=
            [
                logging.FileHandler(file_path, mode='a'),
                logging.StreamHandler()
            ] if not supress_file else 
            [logging.StreamHandler()]
        )
    return logging.getLogger(id) # Unique per instance
    
def get_current_time_str():
    return datetime.now().strftime("%Y%m%d%H%M%S")

def merge_mean(means, sizes):
    '''Function that merge two means of groups with their respective group size given'''
    for i in range(len(means)):
        means[i] = means[i] * sizes[i]
    return sum(means) / sum(sizes)

def merge_std(stds, means, sizes):
    '''Function that merge two stds of groups with their respective group size and mean given'''
    for i in range(len(stds)):
        stds[i] = stds[i] * sizes[i] + (means[i] ** 2) * sizes[i]
    return np.sqrt(sum(stds) / sum(sizes) - (merge_mean(means, sizes) ** 2))

class DummySummaryWriter:
    '''Used to easily disable the summary writer for debug purposes.'''
    def __init__(*args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, *args, **kwargs):
        return self