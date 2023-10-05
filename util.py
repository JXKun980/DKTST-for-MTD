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
    Last sample can be less than the sample size specified
    '''
        
    def __init__(self, 
                 device,
                 sample_size: int,
                 dataset_size: int,
                 sample_count: int = None) -> None:
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
        self.__sample_count = sample_count if sample_count else self.dataset_size // self.sample_size
        # If None, default to sample without replacement, so sample count = dataset_size / sample_size.
        # Else, the sample without replacement procedure is performed multiple times until the desired count is returned.
        
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


class Two_Sample_Dataset(Dataset):
    def __init__(self, 
                 data_s1: 'list[str]', 
                 data_s2: 'list[str]',
                 device,
                 sample_size: int,
                 sample_count: int = None
                 ) -> None:
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
        new_self.data_s1 = f(copy.deepcopy(self.data_s1))
        new_self.data_s2 = f(copy.deepcopy(self.data_s2))
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
            device,
            sample_count_test: int = None,
            ):
        '''Constructor function that construct train and test sets from arguments'''

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
            data_s1_tr_aggregated = np.concatenate((data_s1_tr_aggregated, data_s1_tr[:len(data_s1_tr)//len(datasets)]))
            data_s2_tr_aggregated = np.concatenate((data_s2_tr_aggregated, data_s2_tr[:len(data_s1_tr)//len(datasets)]))
            data_s1_te_aggregated = np.concatenate((data_s1_te_aggregated, data_s1_te[:len(data_s1_te)//len(datasets)]))
            data_s2_te_aggregated = np.concatenate((data_s2_te_aggregated, data_s2_te[:len(data_s1_te)//len(datasets)]))
        
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
                 device,
                 sample_size: int,
                 sample_count: int = None
                 ) -> None:
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
            for _ in range(self.sample_cnt):
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
        device,
        sample_count: int = None,
        ):

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

def merge_mean(means, sizes):
    for i in range(len(means)):
        means[i] = means[i] * sizes[i]
    return sum(means) / sum(sizes)

def merge_std(stds, means, sizes):
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

# class Two_Sample_Dataset_OLD(Dataset):
#     def __init__(self, 
#                  sample_size: int,
#                  data_s1: 'list[str]', 
#                  data_s2: 'list[str]') -> None:
        
#         super().__init__(sample_size)

#         self.data_s1 = data_s1
#         self.data_s2 = data_s2
    
#     def __len__(self):
#         return math.ceil(len(self.data_s1) / self.sample_size)
    
#     def __getitem__(self, idx):
#         lo = idx * self.sample_size
#         hi = min(lo + self.sample_size, len(self.data_s1))
#         s1 = self.data_s1[lo:hi]
#         s2 = self.data_s2[lo:hi]
#         return s1, s2
    
#     def copy_and_map(self, f):
#         return Two_Sample_Dataset(
#             sample_size=self.sample_size,
#             data_s1=f(self.data_s1),
#             data_s2=f(self.data_s2)
#         )
        
#     def copy_with_new_sample_size(self, sample_size: int):
#         return Two_Sample_Dataset(
#             sample_size=sample_size,
#             data_s1=self.data_s1,
#             data_s2=self.data_s2,
#         )
        
#     def copy_with_single_type(self, use_s1: bool) -> Two_Sample_Dataset:
#         '''If dataset contain two types of data, get a copy that contain only one type specified by "use_s1"'''
#         return Two_Sample_Dataset(
#             sample_size=self.sample_size,
#             data_s1=self.data_s1 if use_s1 else self.data_s2,
#             data_s2=self.data_s1 if use_s1 else self.data_s2,
#         )


# class Single_Sample_Dataset_OLD(Dataset):
#     '''
#     The aim of this dataset is only for testing the single-sample capability of trained DKTST-for-MTDs
    
#     To get the fill data, the training data set is randomly sampled.
    
#     Returns: 
#         s1: Contains the test data with some fill data to fill to the "sample size"
#             The amount of test data s1 contains is specified by the test_ratio parameter between 0 and 1.
#         s2: Contains only fill data of size "sample_size"
#     '''
#     def __init__(self, 
#                  sample_size: int, 
#                  data_test: 'list[str]', 
#                  data_fill: 'list[str]', 
#                  test_ratio: float) -> None:
#         # Input validation
#         assert 0 < test_ratio < 1, "Test data ratio must be beween 0 and 1 (exclusive)"
        
#         super().__init__(sample_size)
        
#         self.data_test = data_test
#         self.data_fill = data_fill
#         self.test_ratio = test_ratio
#         self.sample_test_size = round(sample_size * test_ratio)
        
#     def __len__(self):
#         return math.ceil(len(self.data_test) / self.sample_size)
    
#     def __getitem__(self, idx):
#         lo = idx * self.sample_test_size
#         hi = min(lo + self.sample_test_size, len(self.data_test))
        
#         s1_fill_size = self.sample_size - (hi-lo)
#         s1_fill_mask = torch.randint(self.data_fill.shape[0], (s1_fill_size,))
#         s1 = torch.cat((self.data_test[lo:hi, :], self.data_fill[s1_fill_mask, :]), 0)
        
#         s2_fill_mask = torch.randint(self.data_fill.shape[0], (self.sample_size,))
#         s2 = self.data_fill[s2_fill_mask, :]
#         return s1, s2

    
#     def copy_and_map(self, f):
#         return Single_Sample_Dataset(
#             sample_size=self.sample_size,
#             data_test=f(self.data_test),
#             data_fill=f(self.data_fill),
#             test_ratio=self.test_ratio
#         )
        
#     def copy_with_new_sample_size(self, sample_size: int):
#         return Single_Sample_Dataset(
#             sample_size=sample_size,
#             data_test=self.data_test,
#             data_fill=self.data_fill,
#             test_ratio=self.test_ratio
#         )  