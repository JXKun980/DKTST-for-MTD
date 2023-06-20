# -*- coding: utf-8 -*-
"""
Created on Dec 21 14:57:02 2019
@author: Learning Deep Kernels for Two-sample Test
@Implementation of MMD-D in our paper on HDGM dataset

BEFORE USING THIS CODE:
1. This code requires PyTorch 1.1.0, which can be found in
https://pytorch.org/get-started/previous-versions/ (CUDA version is 10.1).
2. Numpy and Sklearn are also required. Users can install
Python via Anaconda (Python 3.7.3) to obtain both packages. Anaconda
can be found in https://www.anaconda.com/distribution/#download-section .
"""
import math
import os

import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import dataset_loader as dataset_loader
from utils_HD import MatConvert, MMDu, TST_MMD_u
    
class DKTST:
    class ModelLatentF(torch.nn.Module):
        """Latent space for both domains."""
        def __init__(self, in_dim, H_dim, out_dim):
            """Init latent features."""
            super(type(self), self).__init__()
            self.restored = False
            self.latent = torch.nn.Sequential(
                torch.nn.Linear(in_dim, H_dim, bias=True),
                torch.nn.Softplus(),
                torch.nn.Linear(H_dim, H_dim, bias=True),
                torch.nn.Softplus(),
                torch.nn.Linear(H_dim, H_dim, bias=True),
                torch.nn.Softplus(),
                torch.nn.Linear(H_dim, out_dim, bias=True),
            )
        def forward(self, input):
            """Forward the LeNet."""
            fealant = self.latent(input)
            return fealant
        
    def __init__(self,
                 latent_size_multi, 
                 device, 
                 dtype,
                 logger) -> None:
        
        self.device = device
        self.dtype = dtype
        self.logger = logger
        self.latent_size_multi = latent_size_multi
        
        # Language Model
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
        
        # Deep Kernel
        self.latent_in_dim = self.encoder.config.dim
        self.latent_H_dim = self.latent_in_dim * latent_size_multi
        self.latent_out_dim = self.latent_in_dim * latent_size_multi
        self.reset_deep_kernel()
        
    def reset_deep_kernel(self):
        # Deep Kernel Parameters
        self.epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), self.device, self.dtype))
        self.epsilonOPT.requires_grad = True
        self.sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * self.latent_in_dim), self.device, self.dtype)
        self.sigmaOPT.requires_grad = True
        self.sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.1), self.device, self.dtype)
        self.sigma0OPT.requires_grad = True
        print(self.epsilonOPT.item())
        
        # Deep Neural Network
        self.latent = self.ModelLatentF(self.latent_in_dim, self.latent_H_dim, self.latent_out_dim).to(self.device)
        
    def get_parameters_list(self):
        return list(self.latent.parameters()) + [self.epsilonOPT] + [self.sigmaOPT] + [self.sigma0OPT]
    
    def encode_sentences(self, s):
        """
        @inputs
            s: batch of sentences to be encoded
        """
        with torch.no_grad():
            batch_data_tokenized = self.tokenizer(s, truncation=True, padding=True, return_tensors='pt').to(self.device)
            batch_data_encoded = self.encoder(**batch_data_tokenized) #dim : [batch_size(nr_sentences), tokens, emb_dim]
            batch_data_cls = batch_data_encoded.last_hidden_state[:,0,:] # First token embedding for each sample is the CLS output
        return batch_data_cls
        
    def train_and_test(self, s1_tr, s1_te, s2_tr, s2_te, lr, n_epoch, batch_size_tr, batch_size_te, save_folder, perm_cnt, sig_level, continue_epoch=0, eval_inteval=100, seed=1102):
        self.logger.debug("Start training...")
        
        # Input validation
        assert len(s1_tr) == len(s2_tr) and len(s1_te) == len(s2_te), "S1 and S2 must contain same number of samples"
        
        # Set up tensorboard
        writer = SummaryWriter(log_dir=save_folder)
        
        # Set fixed seeds
        np.random.seed(seed=seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Create save folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # Init Running Statistics
        J_stars_epoch = np.zeros([n_epoch])
        mmd_values_epoch = np.zeros([n_epoch])
        mmd_stds_epoch = np.zeros([n_epoch])
            
        # Init optimizer
        optimizer = torch.optim.Adam(self.get_parameters_list(), lr=lr)
        
        # Formulate and encode dataset
        train_size = len(s1_tr)
        train_cls_s1 = self.encode_sentences(s1_tr)
        train_cls_s2 = self.encode_sentences(s2_tr)
        
        # Training loop
        batch_cnt = math.ceil(train_size / batch_size_tr)
        best_power = 0
        best_checkpoint = 0
        self.latent.train()
        for t in tqdm(range(continue_epoch, n_epoch), desc="Training Progress"): # Epoch Loop
            J_stars_batch = np.zeros([batch_cnt])
            mmd_values_batch = np.zeros([batch_cnt])
            mmd_stds_batch = np.zeros([batch_cnt])
            for b in range(0, batch_cnt): # Batche Loop
                # Get batch data
                start = b*batch_size_tr
                end = min((b+1)*batch_size_tr, train_size)
                batch_size_actual = end - start
                batch_cls = torch.cat((train_cls_s1[start:end,:], train_cls_s2[start:end,:]))
            
                # Compute epsilon, sigma and sigma_0
                ep = torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))
                sigma = self.sigmaOPT ** 2
                sigma0_u = self.sigma0OPT ** 2
                
                # Compute output of the deep network
                batch_data_latent = self.latent(batch_cls)
                
                # Compute J (STAT_u)
                TEMP = MMDu(batch_data_latent, batch_size_actual, batch_cls, sigma, sigma0_u, ep)
                mmd_value_temp = -1 * (TEMP[0]+10**(-8))
                mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
                if mmd_std_temp.item() == 0:
                    self.logger.error('error!!')
                if np.isnan(mmd_std_temp.item()):
                    self.logger.error('error!!')
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                
                # Backprop
                optimizer.zero_grad()
                STAT_u.backward(retain_graph=True)
                optimizer.step()
                
                # Record stats for the batch
                J_stars_batch[b] = STAT_u.item()
                mmd_values_batch[b] = mmd_value_temp.item()
                mmd_stds_batch[b] = mmd_std_temp.item()
                
            # Calculate epoch stats
            J_star_epoch = np.average(J_stars_batch)
            mmd_value_epoch = np.average(mmd_values_batch)
            mmd_std_epoch = np.average(mmd_stds_batch)
            train_power, train_threshold, train_mmd = self.test(s1=s1_tr, s2=s2_tr, batch_size=batch_size_tr, perm_cnt=perm_cnt, sig_level=sig_level) # Training accuracy
            
            # Record epoch stats
            J_stars_epoch[t] = J_star_epoch
            mmd_values_epoch[t] = mmd_value_epoch
            mmd_stds_epoch[t] = mmd_std_epoch
            writer.add_scalar('Train/J_stars', J_star_epoch, t)
            writer.add_scalar('Train/mmd_values', mmd_value_epoch, t)
            writer.add_scalar('Train/mmd_stds', mmd_std_epoch, t)
            writer.add_scalar('Train/train_power', train_power, t)
            writer.add_scalar('Train/train_threshold', train_threshold, t)
            writer.add_scalar('Train/train_mmd', train_mmd, t)
            self.logger.info(f"Epoch {t}: "
                  + f"mmd_value: {1 * J_star_epoch:0>.8} " 
                  + f"mmd_std: {mmd_value_epoch:0>.8} "
                  + f"Statistic: {-1 * mmd_std_epoch:0>.8} "
                  + f"train_power: {train_power} "
                  + f"train_threshold: {train_threshold} "
                  + f"train_mmd: {train_mmd} ")
            
            # At checkpoint
            if t != 0 and t % eval_inteval == 0:
                self.logger.info(f"========== Checkpoint at Epoch {t} ==========")
                
                # Evaluate model using test set
                self.logger.info("Validating model...")
                
                # Custom test consists of both type 1 and 2 error test, for different batch sizes
                val_power_avg = self.custom_test_procedure(s1_te, s2_te, perm_cnt, sig_level, writer, t)
                
                # Update best model
                if val_power_avg >= best_power:
                    best_power = val_power_avg
                    best_checkpoint = t
                    self.logger.info(f"Best model updated")
            
                # Save model
                file_path = f'{save_folder}/model_ep_{t}.pth'
                self.logger.info(f"Saving model to {file_path}...")
                torch.save(self.latent.state_dict(), file_path)
            
        # Save deep kernel neural network
        file_path = f'{save_folder}/model_ep_{t}.pth'
        self.logger.info(f"Saving model to {file_path}...")
        torch.save(self.latent.state_dict(), file_path)
        
        # Indicate best model
        self.logger.info(f"Best model at checkpoint {best_checkpoint}")
        
        return J_stars_epoch, mmd_values_epoch, mmd_stds_epoch
    
    def load(self, model_path):
        self.latent.load_state_dict(torch.load(model_path))
        
    def test(self, s1, s2, batch_size, perm_cnt, sig_level, seed=1102):
        self.logger.debug("Start testing...")
        self.latent.eval()
        
        # Input validation
        assert len(s1) == len(s2), "S1 and S2 must contain same number of samples"
        
        # Set fixed seeds
        np.random.seed(seed=seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Formulate and encode dataset
        test_size = len(s1)
        test_cls_s1 = self.encode_sentences(s1)
        test_cls_s2 = self.encode_sentences(s2)
        
        # Init test stats
        count_u = 0
        batch_cnt = math.ceil(test_size / batch_size) # Assuming dataset contains integer number of batches
        H_u = np.zeros(batch_cnt)
        T_u = np.zeros(batch_cnt)
        M_u = np.zeros(batch_cnt)
        
        # Test loop over data batches
        for b in range(0, batch_cnt):
            # Get batch data
            start = b*batch_size
            end = min((b+1)*batch_size, test_size)
            batch_size_actual = end - start
            batch_cls = torch.cat((test_cls_s1[start:end,:], test_cls_s2[start:end,:]))
            
            # Compute epsilon, sigma and sigma_0
            ep = torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))
            sigma = self.sigmaOPT ** 2
            sigma0_u = self.sigma0OPT ** 2
            
            # Get output from deep kernel using two sample test
            h_u, threshold_u, mmd_value_u = TST_MMD_u(
                self.latent(batch_cls), 
                perm_cnt, 
                batch_size_actual, 
                batch_cls, 
                sigma, 
                sigma0_u, 
                ep, 
                sig_level, 
                self.device, 
                self.dtype)

            # Gather results
            count_u = count_u + h_u
            # print("MMD-DK:", count_u)
            H_u[b] = h_u
            T_u[b] = threshold_u
            M_u[b] = mmd_value_u
            
        test_power = H_u.sum() / float(batch_cnt)
        threshold_avg = np.average(T_u)
        mmd_avg = np.average(M_u)
        
        self.latent.train()
        return test_power, threshold_avg, mmd_avg
    
    def custom_test_procedure(self, s1, s2, perm_cnt, sig_level, writer=None, epoch=None):
        batch_sizes = [10, 5, 4, 3]
        val_power_diff_sum = 0
        for bs in batch_sizes: # Test for all these batch sizes for testing
            # Test of Type II error (Assuming s1 and s2 are different distributions)
            val_power1, val_threshold1, val_mmd1 = self.test(s1=s1, s2=s2, batch_size=bs, perm_cnt=perm_cnt, sig_level=sig_level) 
            # Test of Type I error (Assuming s1 and s2 are different distributions)
            val_power2, val_threshold2, val_mmd2 = self.test(s1=s1, s2=s1, batch_size=bs, perm_cnt=perm_cnt, sig_level=sig_level) 
            
            val_power_diff_sum += val_power1
            
            if writer and epoch:
                writer.add_scalar(f'Validation/val_power_diff_{bs}', val_power1, epoch)
                writer.add_scalar(f'Validation/val_threshold_diff_{bs}', val_threshold1, epoch)
                writer.add_scalar(f'Validation/val_mmd_diff_{bs}', val_mmd1, epoch)
            self.logger.info(f"Validation power (Diff) batch size {bs}: {val_power1} "
                        + f"Validation threshold (Diff) batch size {bs}: {val_threshold1} "
                        + f"Validation mmd (Diff) batch size {bs}: {val_mmd1} ")
            
            if writer and epoch:
                writer.add_scalar(f'Validation/val_power_same_{bs}', val_power2, epoch)
                writer.add_scalar(f'Validation/val_threshold_same_{bs}', val_threshold2, epoch)
                writer.add_scalar(f'Validation/val_mmd_same_{bs}', val_mmd2, epoch)
            self.logger.info(f"Validation power (Same) batch size {bs}: {val_power2} "
                        + f"Validation threshold (Same) batch size {bs}: {val_threshold2} "
                        + f"Validation mmd (Same) batch size {bs}: {val_mmd2} ")
        val_power_avg = val_power_diff_sum / len(batch_sizes)
        
        return val_power_avg
