import random
from datetime import datetime
import logging

import numpy as np
import torch

from external import dataset_loader

def load_data(dataset_name, llm_name, shuffle=False, train_ratio=0.8):
    data = dataset_loader.load(name=dataset_name, detectLLM=llm_name, train_ratio=train_ratio)
    
    assert len(data) % 2 == 0, "Dataset must contain pairs of human and machine answers, i.e. {Human, Machine, Human...}"
    
    data_human_tr = data['train']['text'][0::2]
    data_machine_tr = data['train']['text'][1::2]
    data_human_te = data['test']['text'][0::2]
    data_machine_te = data['test']['text'][1::2]
    
    # Random shuffle the dataset so each pair of answers do not correspond to the same questions
    if shuffle:
        random.shuffle(data_human_tr)
        random.shuffle(data_machine_tr)
        random.shuffle(data_human_te)
        random.shuffle(data_machine_te)
    
    return data_human_tr, data_human_te, data_machine_tr, data_machine_te

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