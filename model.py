import math
import os

import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from external.dktst_utils_HD import MatConvert, MMDu, TST_MMD_u
import util
    
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
        self.latent = self.ModelLatentF(self.latent_in_dim, self.latent_H_dim, 
                                        self.latent_out_dim).to(self.device)
        
        # Training
        self.optimizer = torch.optim.Adam(self.get_parameters_list())
        
    def get_parameters_list(self):
        return list(self.latent.parameters()) + [self.epsilonOPT] + [self.sigmaOPT] + [self.sigma0OPT]
    
    def encode_sentences(self, s):
        """
        @inputs
            s: batch of sentences to be encoded
        """
        with torch.no_grad():
            batch_data_tokenized = self.tokenizer(s, truncation=True, padding=True, return_tensors='pt').to(self.device)
            # truncation=True: truncate to the maximum length the model is allowed to take
            # padding=True: pad to the longest sequence of the batch
            batch_data_encoded = self.encoder(**batch_data_tokenized) #dim : [batch_size(nr_sentences), tokens, emb_dim]
            batch_data_cls = batch_data_encoded.last_hidden_state[:,0,:] # First token embedding for each sample is the CLS output
        return batch_data_cls
        
    def train_and_test(self, s1_tr, s1_te, s2_tr, s2_te, lr, n_epoch, batch_size_tr, 
                       batch_size_te, save_folder, perm_cnt, sig_lvl, start_epoch=0, 
                       use_custom_test=True, eval_inteval=100, save_interval=500, seed=1102):
        self.logger.debug("Start training...")
        
        n_epoch += 1 # To end up at a whole numbered epoch
        
        # Set up tensorboard
        writer = SummaryWriter(log_dir=save_folder)
        
        # Set fixed seeds
        util.setup_seeds(seed=seed)
        
        # Create save folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        # Set up optmizer
        if start_epoch == 0: # if starting fresh training, use new LR
            self.optimizer = torch.optim.Adam(self.get_parameters_list(), lr=lr)
        
        # Init Running Statistics
        J_stars_epoch = np.zeros([n_epoch])
        mmd_values_epoch = np.zeros([n_epoch])
        mmd_stds_epoch = np.zeros([n_epoch])
        
        # Formulate and encode dataset
        train_size = len(s1_tr)
        train_cls_s1 = self.encode_sentences(s1_tr)
        train_cls_s2 = self.encode_sentences(s2_tr)
        
        # Training loop
        batch_cnt = math.ceil(train_size / batch_size_tr)
        best_power = 0
        best_chkpnt = 0
        self.latent.train()
        for t in tqdm(range(start_epoch, n_epoch), desc="Training Progress"): # Epoch Loop, +1 here to end at a whole numbered epoch
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
                self.optimizer.zero_grad()
                STAT_u.backward(retain_graph=True)
                self.optimizer.step()
                
                # Record stats for the batch
                J_stars_batch[b] = STAT_u.item()
                mmd_values_batch[b] = mmd_value_temp.item()
                mmd_stds_batch[b] = mmd_std_temp.item()
                
            # Calculate epoch stats
            J_star_epoch = np.average(J_stars_batch)
            mmd_value_epoch = np.average(mmd_values_batch)
            mmd_std_epoch = np.average(mmd_stds_batch)

            # Record epoch stats
            J_stars_epoch[t] = J_star_epoch
            mmd_values_epoch[t] = mmd_value_epoch
            mmd_stds_epoch[t] = mmd_std_epoch
            writer.add_scalar('Train/J_stars', J_star_epoch, t)
            writer.add_scalar('Train/mmd_values', mmd_value_epoch, t)
            writer.add_scalar('Train/mmd_stds', mmd_std_epoch, t)
            self.logger.info(f"Epoch {t}: "
                  + f"mmd_value: {1 * J_star_epoch:0>.8} " 
                  + f"mmd_std: {mmd_value_epoch:0>.8} "
                  + f"Statistic: {-1 * mmd_std_epoch:0>.8} ")
            
            # At checkpoint
            if t != 0:
                if t % eval_inteval == 0:
                    self.logger.info(f"========== Checkpoint at Epoch {t} ==========")
                    
                    # Evaluate model using training set
                    self.logger.info("Recording training accuracy...")
                    train_power, train_threshold, train_mmd = self.test(s1=s1_tr, s2=s2_tr, batch_size=batch_size_te, 
                                                        perm_cnt=perm_cnt, sig_lvl=sig_lvl)
                    writer.add_scalar('Train/train_power', train_power, t)
                    writer.add_scalar('Train/train_threshold', train_threshold, t)
                    writer.add_scalar('Train/train_mmd', train_mmd, t)
                    self.logger.info(f"Epoch {t}: "
                    + f"train_power: {train_power} "
                    + f"train_threshold: {train_threshold} "
                    + f"train_mmd: {train_mmd} ")
                    
                    # Evaluate model using test set
                    self.logger.info("Validating model...")
                    
                    # Custom test consists of both type 1 and 2 error test, for different batch sizes
                    if use_custom_test:
                        val_power_avg = self.custom_test_procedure(s1_te, s2_te, perm_cnt, sig_lvl, writer, t)
                    else:
                        val_power_avg, _, _ = self.test(s1_te, s2_te, batch_size_te, perm_cnt, sig_lvl, seed)
                        # Note no validation result put into running statistics for tensorboard TODO
                    
                    # Update best model
                    if val_power_avg >= best_power:
                        best_power = val_power_avg
                        best_chkpnt = t
                    self.logger.info(f"Best model at {best_chkpnt}")
                if t % save_interval == 0:
                    self.save(save_folder=save_folder, epoch=t)
            
        # Save deep kernel neural network
        self.save(save_folder=save_folder, epoch=t)
        
        # Indicate best model
        self.logger.info(f"Best model at checkpoint {best_chkpnt}")
        
        return J_stars_epoch, mmd_values_epoch, mmd_stds_epoch
    
    def load(self, chkpnt_path):
        chkpnt = torch.load(chkpnt_path)
        self.latent.load_state_dict(chkpnt['model_state_dict'])
        self.optimizer.load_state_dict(chkpnt['optimizer_state_dict'])
        
    def save(self, save_folder, epoch):
        file_path = f'{save_folder}/model_ep_{epoch}.pth'
        self.logger.info(f"Saving model to {file_path}...")
        chkpnt = {
            'model_state_dict': self.latent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(chkpnt, file_path)
        
    def test(self, s1, s2, batch_size, perm_cnt, sig_lvl, seed=1102):
        '''
        Note test only test full batches, that is if batch size is larger than test size, that part will be discarded.
        Therefore you need to specify a small test batch size (such as 20 or 10)
        '''
        self.logger.debug("Start testing...")
        self.latent.eval()
        
        # Input validation
        assert len(s1) == len(s2), "S1 and S2 must contain same number of samples"
        
        # Set fixed seeds
        util.setup_seeds(seed=seed)
        
        # Formulate and encode dataset
        test_size = len(s1)
        self.logger.info(f"Test size is {test_size}")
        test_cls_s1 = self.encode_sentences(s1)
        test_cls_s2 = self.encode_sentences(s2)
        
        # Init test stats
        count_u = 0
        batch_cnt = test_size // batch_size # Assuming dataset contains integer number of batches, otherwise truncate
        H_u = np.zeros(batch_cnt)
        T_u = np.zeros(batch_cnt)
        M_u = np.zeros(batch_cnt)
        
        # Test loop over data batches
        for b in range(0, batch_cnt):
            # Get batch data
            start = b*batch_size
            end = (b+1)*batch_size
            batch_cls = torch.cat((test_cls_s1[start:end,:], test_cls_s2[start:end,:]))
            
            # Compute epsilon, sigma and sigma_0
            ep = torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))
            sigma = self.sigmaOPT ** 2
            sigma0_u = self.sigma0OPT ** 2
            
            # Get output from deep kernel using two sample test
            h_u, threshold_u, mmd_value_u = TST_MMD_u(
                self.latent(batch_cls), 
                perm_cnt, 
                batch_size, 
                batch_cls, 
                sigma, 
                sigma0_u, 
                ep, 
                sig_lvl, 
                self.device, 
                self.dtype)

            # Gather results
            count_u = count_u + h_u
            # print("MMD-DK:", count_u)
            H_u[b] = h_u
            T_u[b] = threshold_u
            M_u[b] = mmd_value_u
            
        test_power = H_u.sum() / float(batch_cnt)
        threshold_avg = np.average(T_u[T_u != np.nan])
        mmd_avg = np.average(M_u)
        
        self.latent.train()
        return test_power, threshold_avg, mmd_avg
    
    def custom_test_procedure(self, s1, s2, perm_cnt, sig_lvl, writer=None, epoch=None):
        batch_sizes = [10]
        val_power_diff_sum = 0
        for bs in batch_sizes: # Test for all these batch sizes for testing
            # Test of Type II error (Assuming s1 and s2 are different distributions)
            val_power1, val_threshold1, val_mmd1 = self.test(s1=s1, s2=s2, batch_size=bs, perm_cnt=perm_cnt, sig_lvl=sig_lvl) 
            # Test of Type I error (Assuming s1 and s2 are different distributions)
            val_power2, val_threshold2, val_mmd2 = self.test(s1=s1, s2=s1, batch_size=bs, perm_cnt=perm_cnt, sig_lvl=sig_lvl) 
            
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
