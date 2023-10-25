import os
import logging
import copy

import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from external.dktst_utils_HD import MatConvert, MMDu, TST_MMD_u
import util
    
class DKTST_for_MTD:
    '''
    The main class for the DKTST-for-MTD model
    '''
    
    class ModelLatentF(torch.nn.Module):
        """Linear layers for latent features."""
        
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
            self.latent.apply(self.weight_init)
        
        def forward(self, input):
            """Forward the LeNet."""
            fealant = self.latent(input)
            return fealant
        
        def weight_init(self, m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        
    def __init__(self,
                 latent_size_multi: int, 
                 device: str, 
                 dtype: torch.dtype, # torch.float32 or torch.float64
                 logger: logging.Logger,
                 debug: bool=False) -> None:
        
        self.device = device
        self.dtype = dtype
        self.logger = logger
        self.debug = debug
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
        
        # Deep Neural Network
        self.latent = self.ModelLatentF(self.latent_in_dim, self.latent_H_dim, 
                                        self.latent_out_dim).to(self.device)
        
        # Training
        self.optimizer = torch.optim.Adam(self.get_parameters_list())
        
    def get_parameters_list(self):
        return list(self.latent.parameters()) + [self.epsilonOPT] + [self.sigmaOPT] + [self.sigma0OPT]
    
    def encode_sentences(self, s):
        """
        @params
            s: Sentences to be encoded. np.NDArray[str] of shape (sample_size,) 
        @returns
            np.array[float] of shape (sample_size, encoding_size)
        """
        with torch.no_grad():
            batch_data_tokenized = self.tokenizer(s, truncation=True, padding=True, return_tensors='pt').to(self.device)
            # truncation=True: truncate to the maximum length the model is allowed to take
            # padding=True: pad to the longest sequence of the batch
            batch_data_encoded = self.encoder(**batch_data_tokenized) #dim : [sample_size(nr_sentences), tokens, emb_dim]
            batch_data_cls = batch_data_encoded.last_hidden_state.detach().cpu()[:,0,:].numpy() # First token embedding for each sample is the CLS output
        return batch_data_cls
    
    def get_J_star(self, s1s2_cls):
        """
        @params
            s1s2_cls: Numpy array of sentence encodings (encoded CLS tokens). np.array[float] of shape (sample_size, encoding_size)
        @returns
            STAT_u: J_star estimate
            mmd_value_temp: MMD value estimate
            mmd_std_temp: MMD standard deviation estimate
        """
        # Compute epsilon, sigma and sigma_0
        ep = torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))
        sigma = self.sigmaOPT ** 2
        sigma0_u = self.sigma0OPT ** 2
                
        # Compute output of the deep network
        s1s2_latent = self.latent(s1s2_cls)
        
        # Compute J (STAT_u)
        TEMP = MMDu(s1s2_latent, s1s2_cls.shape[0]//2, s1s2_cls, sigma, sigma0_u, ep)
        mmd_value_temp = -1 * (TEMP[0]+10**(-8))
        mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
        if mmd_std_temp.item() == 0:
            self.logger.error('Error: mmd standard deviation is 0')
            # raise util.TrainingError('Error: mmd standard deviation is 0')
        if np.isnan(mmd_std_temp.item()):
            self.logger.error('Error: mmd standard deviation is nan')
            # raise util.TrainingError('Error: mmd standard deviation is nan')
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        return STAT_u, mmd_value_temp, mmd_std_temp
        
    def train_and_test(self, 
                       data_tr: util.Two_Sample_Dataset, 
                       data_te: util.Two_Sample_Dataset, 
                       lr: float, 
                       total_epoch: int, 
                       save_folder: str, 
                       perm_cnt: int, 
                       sig_lvl: float, 
                       start_epoch: int=0, 
                       eval_inteval: int=100, 
                       save_interval: int=500, 
                       seed: int=1103):
        """
        Main training procedure (with validation testing)
        
        @params
            data_tr: Training dataset
            data_te: Test (Validation) dataset
            lr: Initial learning rate for Adam optimizer
            total_epoch: Total number of epochs to train
            save_folder: Folder name (in current working directory) to save checkpoints
            perm_cnt: Number of permutations to use for two sample test
            sig_lvl: Significance level for two sample test
            start_epoch: Epoch to start training from (useful for resuming training)
            eval_inteval: Number of epochs between each evaluation
            save_interval: Number of epochs between each checkpoint save
            seed: Seed to use for training
        @returns
            J_stars_epoch: List of J_star estimates for each epoch
            mmd_values_epoch: List of MMD value estimates for each epoch
            mmd_stds_epoch: List of MMD standard deviation estimates for each epoch
        """
        self.logger.debug("Start training...")
        
        total_epoch += 1 # To end up at a whole numbered epoch
        
        # Set up tensorboard
        if not self.debug:
            writer = SummaryWriter(log_dir=save_folder)
        else:
            writer = util.DummySummaryWriter()
            logging.debug("Tensorboard file not written due to debug mode.")
        
        # Set fixed seeds
        util.setup_seeds(seed=seed)
        
        # Create save folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        # Set up optmizer
        if start_epoch == 0: # if starting fresh training, use new LR
            self.optimizer = torch.optim.Adam(self.get_parameters_list(), lr=lr)
        
        # Init Running Statistics
        J_stars_epoch = np.zeros([total_epoch])
        mmd_values_epoch = np.zeros([total_epoch])
        mmd_stds_epoch = np.zeros([total_epoch])
        
        # Formulate and encode dataset
        data_tr_cls = data_tr.copy_and_map(lambda a: self.encode_sentences(a.tolist()))
        
        # Get type 1 error test data
        data_te_same = data_te.copy_with_single_type(use_s1=True)
        
        # Training loop
        self.latent.train()
        for t in tqdm(range(start_epoch, total_epoch), initial=start_epoch, total=total_epoch, desc="Training Progress"): # Epoch Loop, +1 here to end at a whole numbered epoch
            J_stars_batch = np.zeros([len(data_tr_cls)])
            mmd_values_batch = np.zeros([len(data_tr_cls)])
            mmd_stds_batch = np.zeros([len(data_tr_cls)])
            
            for s_idx, (s1_cls, s2_cls) in enumerate(data_tr_cls): # Batch Loop
                # Get batch data
                s1_cls = torch.squeeze(s1_cls, 0)
                s2_cls = torch.squeeze(s2_cls, 0)
                s1s2_cls = torch.cat((s1_cls, s2_cls))
            
                STAT_u, mmd_value_temp, mmd_std_temp = self.get_J_star(s1s2_cls)
                
                # Backprop
                self.optimizer.zero_grad()
                STAT_u.backward(retain_graph=True)
                self.optimizer.step()
                
                # Record stats for the batch
                J_stars_batch[s_idx] = STAT_u.item()
                mmd_values_batch[s_idx] = mmd_value_temp.item()
                mmd_stds_batch[s_idx] = mmd_std_temp.item()
                
            # Calculate epoch stats
            J_star_epoch = J_stars_batch.mean()
            mmd_value_epoch = mmd_values_batch.mean()
            mmd_std_epoch = mmd_stds_batch.mean()

            # Record epoch stats
            J_stars_epoch[t] = J_star_epoch
            mmd_values_epoch[t] = mmd_value_epoch
            mmd_stds_epoch[t] = mmd_std_epoch
            writer.add_scalar('Train/J_stars', J_star_epoch, t)
            writer.add_scalar('Train/mmd_values', mmd_value_epoch, t)
            writer.add_scalar('Train/mmd_stds', mmd_std_epoch, t)
            self.logger.info(f"Epoch {t}: "
                  + f"J_stars: {1 * J_star_epoch:0>.8} " 
                  + f"mmd_values: {mmd_value_epoch:0>.8} "
                  + f"mmd_stds: {-1 * mmd_std_epoch:0>.8} ")
            
            # At checkpoint
            if t != 0:
                if t % eval_inteval == 0:
                    self.logger.info(f"========== Checkpoint at Epoch {t} ==========")
                    
                    # Evaluate model using training set
                    self.logger.info("Recording training accuracy...")
                    data_tr_eval = copy.copy(data_tr)
                    data_tr_eval.sample_size = data_te.sample_size
                    data_tr_eval.sample_count = data_te.sample_count
                    train_power, train_thresholds, train_mmds = self.test(
                                                        data=data_tr_eval,
                                                        perm_cnt=perm_cnt, 
                                                        sig_lvl=sig_lvl)
                    train_thresholds_mean = train_thresholds.mean()
                    train_mmds_mean = train_mmds.mean()
                    writer.add_scalar('Train/train_power', train_power, t)
                    writer.add_scalar('Train/train_threshold_mean', train_thresholds_mean, t)
                    writer.add_scalar('Train/train_mmd_mean', train_mmds_mean, t)
                    self.logger.info(f"Epoch {t}: "
                    + f"train_power: {train_power} "
                    + f"train_threshold_mean: {train_thresholds_mean} "
                    + f"train_mmd_mean: {train_mmds_mean} ")
                    
                    # Evaluate model using test set
                    self.logger.info("Validating model...")
                    
                    _, _ = self.validation( data_diff=data_te, 
                                            data_same=data_te_same, 
                                            perm_cnt=perm_cnt, 
                                            sig_lvl=sig_lvl, 
                                            writer=writer, 
                                            epoch=t)
                    
                if t % save_interval == 0:
                    self.save(save_folder=save_folder, epoch=t)
            
        # Save deep kernel neural network
        self.save(save_folder=save_folder, epoch=t)
        
        return J_stars_epoch, mmd_values_epoch, mmd_stds_epoch

    def load(self, chkpnt_path: str):
        '''
        Load checkpoint for the model
        
        @params
            chkpnt_path: Path to checkpoint file
        '''
        chkpnt = torch.load(chkpnt_path)
        self.latent.load_state_dict(chkpnt['model_state_dict'])
        self.optimizer.load_state_dict(chkpnt['optimizer_state_dict'])
        
        self.epsilonOPT = chkpnt['epsilonOPT'].to(self.device)
        self.sigmaOPT = chkpnt['sigmaOPT'].to(self.device)
        self.sigma0OPT = chkpnt['sigma0OPT'].to(self.device)
        
    def save(self, save_folder: str, epoch: int):
        '''
        Save checkpoint for the model
        
        @params
            save_folder: Folder name (in current working directory) to save checkpoints
            epoch: Epoch number to save checkpoint at
        '''
        if not self.debug:
            file_path = f'{save_folder}/model_ep_{epoch}.pth'
            self.logger.info(f"Saving model to {file_path}...")
            chkpnt = {
                'model_state_dict': self.latent.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilonOPT': self.epsilonOPT,
                'sigmaOPT': self.sigmaOPT,
                'sigma0OPT': self.sigma0OPT
            }
            torch.save(chkpnt, file_path)
        else:
            logging.debug("Model not saved due to debug mode.")
        
    def test(self, data: util.Dataset, perm_cnt: int, sig_lvl: float, seed: int=1103):
        '''
        Main test procedure
        
        @params
            data: Dataset to test on
            perm_cnt: Number of permutations to use for two sample test
            sig_lvl: Significance level for two sample test
            seed: Seed to use for testing
        @returns
            test_power: Test power
            thresholds: List of thresholds for each test sample
            mmds: List of MMD values for each test sample
        '''
        self.logger.debug("Start testing...")
        self.latent.eval()
        with torch.no_grad():
            # Set fixed seeds
            util.setup_seeds(seed=seed)
            
            # Formulate and encode dataset
            self.logger.info(f"Test size is {len(data)}")
            data_cls = data.copy_and_map(lambda a: self.encode_sentences(a.tolist()))
            
            # Init test stats
            count_u = 0
            H_u = np.zeros(len(data_cls))
            T_u = np.zeros(len(data_cls))
            M_u = np.zeros(len(data_cls))
            
            # Test loop over data batches
            for s_idx, (s1_cls, s2_cls) in enumerate(data_cls):
                # Get batch data
                s1_cls = torch.squeeze(s1_cls, 0)
                s2_cls = torch.squeeze(s2_cls, 0)
                s1s2_cls = torch.cat((s1_cls, s2_cls))
                
                # Compute epsilon, sigma and sigma_0
                ep = torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))
                sigma = self.sigmaOPT ** 2
                sigma0_u = self.sigma0OPT ** 2
                
                # Get output from deep kernel using two sample test
                h_u, threshold_u, mmd_value_u = TST_MMD_u(
                    self.latent(s1s2_cls), 
                    perm_cnt, 
                    s1_cls.shape[0], 
                    s1s2_cls, 
                    sigma, 
                    sigma0_u, 
                    ep, 
                    sig_lvl, 
                    self.device, 
                    self.dtype)

                # Gather results
                count_u = count_u + h_u
                # print("MMD-DK:", count_u)
                H_u[s_idx] = h_u
                T_u[s_idx] = threshold_u
                M_u[s_idx] = mmd_value_u
                
            test_power = H_u.mean()
            thresholds = T_u[~np.isnan(T_u)]
            mmds = M_u

            self.latent.train()
            
        return test_power, thresholds, mmds
    
    def strong_single_sample_test(self, data: util.Dataset, data_comp: util.Dataset, perm_cnt: int, sig_lvl: float, seed: int=1103):
        '''
        Stronger single sample test that requires two datasets, each contains fill data with a different data type (human and machine).
        Structure is largely similar to test() above.
        Each test sample set is tested twice, against both types of fill data. If results from the two tests do not agree, the test procedure is repeated until they agree.
        
        @params
            data: Dataset to test on
            data_comp: Dataset to test on (with different data type)
            perm_cnt: Number of permutations to use for two sample test
            sig_lvl: Significance level for two sample test
            seed: Seed to use for testing
        '''
        self.logger.debug("Start strong single sample testing...")
        self.latent.eval()
        with torch.no_grad():
            # Set fixed seeds
            util.setup_seeds(seed=seed)
            
            # Formulate and encode dataset
            self.logger.info(f"Test size is {len(data)}")
            data_cls = data.copy_and_map(lambda a: self.encode_sentences(a.tolist()))
            data_comp_cls = data_comp.copy_and_map(lambda a: self.encode_sentences(a.tolist()))
            
            # Init test stats
            count_u = 0
            H_u = np.zeros(len(data_cls))
            
            # Test loop over data batches
            data_cls_iter = iter(data_cls)
            data_comp_cls_iter = iter(data_comp_cls)
            for s_idx in range(len(data_cls)):
                (s1_cls, s2_cls) = data_cls_iter.__next__()
                (s1_comp_cls, s2_comp_cls) = data_comp_cls_iter.__next__()
                
                # Get batch data
                s1_cls = torch.squeeze(s1_cls, 0)
                s2_cls = torch.squeeze(s2_cls, 0)
                s1s2_cls = torch.cat((s1_cls, s2_cls))
                s1_comp_cls = torch.squeeze(s1_comp_cls, 0)
                s2_comp_cls = torch.squeeze(s2_comp_cls, 0)
                s1s2_comp_cls = torch.cat((s1_comp_cls, s2_comp_cls))
                
                # Compute epsilon, sigma and sigma_0
                ep = torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))
                sigma = self.sigmaOPT ** 2
                sigma0_u = self.sigma0OPT ** 2
                
                # Get output from deep kernel using two sample test
                restart_cnt = 0
                while restart_cnt < 10:
                    h_u, _, _ = TST_MMD_u(
                        self.latent(s1s2_cls), 
                        perm_cnt, 
                        s1_cls.shape[0], 
                        s1s2_cls, 
                        sigma, 
                        sigma0_u, 
                        ep, 
                        sig_lvl, 
                        self.device, 
                        self.dtype)
                    h_u_comp, _, _ = TST_MMD_u(
                        self.latent(s1s2_comp_cls), 
                        perm_cnt, 
                        s1_cls.shape[0], 
                        s1s2_cls, 
                        sigma, 
                        sigma0_u, 
                        ep, 
                        sig_lvl, 
                        self.device, 
                        self.dtype)
                    if h_u != h_u_comp: # When results agree
                        break
                    else:
                        restart_cnt += 1
                        util.setup_seeds(seed=seed+restart_cnt)
                        print(f'Restart {restart_cnt}')
                        # Restart test with a different seed
                        
                util.setup_seeds(seed=seed) # Reset seed to initial value

                # Gather results
                count_u = count_u + h_u
                # print("MMD-DK:", count_u)
                H_u[s_idx] = h_u
                
            test_power = H_u.mean()

            self.latent.train()
            
        return test_power
        
    
    def validation( self, 
                    data_diff: util.Two_Sample_Dataset, 
                    data_same: util.Two_Sample_Dataset, 
                    perm_cnt: int, 
                    sig_lvl: float, 
                    writer=None, 
                    epoch: int=None):
        '''
        Main validation procedure
        
        @params
            data_diff: Dataset to test on (with different data type)
            data_same: Dataset to test on (with same data type)
            perm_cnt: Number of permutations to use for two sample test
            sig_lvl: Significance level for two sample test
            writer: Tensorboard writer
            epoch: Epoch number to save checkpoint at
        '''
        sample_sizes = [10]
        val_powers1 = np.empty(len(sample_sizes))
        val_powers2 = np.empty(len(sample_sizes))
        
        for i, ss in enumerate(sample_sizes): # Test for all these batch sizes for testing
            data_diff = copy.copy(data_diff)
            data_diff.sample_size = ss
            data_same = copy.copy(data_same)
            data_same.sample_size = ss
            # Test of Type I error (Assuming s1 and s2 are the same distributions)
            val_power1, val_thresholds1, val_mmds1 = self.test(data=data_same, perm_cnt=perm_cnt, sig_lvl=sig_lvl) 
            # Test of Type II error (Assuming s1 and s2 are different distributions)
            val_power2, val_thresholds2, val_mmds2 = self.test(data=data_diff, perm_cnt=perm_cnt, sig_lvl=sig_lvl) 
            
            val_powers1[i] = val_powers1
            val_powers2[i] = val_powers2
            
            val_threshold_mean1 = val_thresholds1.mean()
            val_threshold_mean2 = val_thresholds2.mean()
            
            val_mmd_mean1 = val_mmds1.mean()
            val_mmd_mean2 = val_mmds2.mean()
           
            if writer and epoch:
                writer.add_scalar(f'Validation/val_power_same_{ss}', val_power1, epoch)
                writer.add_scalar(f'Validation/val_threshold_mean_same_{ss}', val_threshold_mean1, epoch)
                writer.add_scalar(f'Validation/val_mmd_mean_same_{ss}', val_mmd_mean1, epoch)
            self.logger.info(f"Validation power (Same) batch size {ss}: {val_power1} "
                        + f"Validation threshold mean (Same) batch size {ss}: {val_threshold_mean1} "
                        + f"Validation mmd mean (Same) batch size {ss}: {val_mmd_mean1} ")
            
            if writer and epoch:
                writer.add_scalar(f'Validation/val_power_diff_{ss}', val_power2, epoch)
                writer.add_scalar(f'Validation/val_threshold_mean_diff_{ss}', val_threshold_mean2, epoch)
                writer.add_scalar(f'Validation/val_mmd_mean_diff_{ss}', val_mmd_mean2, epoch)
            self.logger.info(f"Validation power (Diff) batch size {ss}: {val_power2} "
                        + f"Validation threshold mean (Diff) batch size {ss}: {val_threshold_mean2} "
                        + f"Validation mmd std (Diff) batch size {ss}: {val_mmd_mean2} ")
        
        return val_powers1, val_powers2
