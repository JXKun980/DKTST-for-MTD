import os
import itertools as it
import abc
from argparse import ArgumentParser

import pandas as pd
import numpy as np

from external.dataset_loader import DATASETS
import util

CSV_NAME_TO_VAR_NAME = {
    'Train - Model Name': 'model_name',
    'Train - Linear Layer Size Multiple': 'linear_size',
    'Train - Dataset Name': 'tr_ds_name',
    'Train - Dataset LLM Name': 'tr_ds_llm_name',
    'Train - S1 S2 Type': 'tr_s1s2',
    'Train - Shuffled': 'tr_shuffled',
    'Train - Epoch Count': 'tr_epo_cnt',
    'Train - Sample Size': 'tr_sample_size',
    'Train - Learning Rate': 'lr',
    'Train - Checkpoint Epoch': 'chkpnt_ep',
    'Train - Seed': 'tr_seed',
    'Test - Dataset Name': 'te_ds_name',
    'Test - Dataset LLM Name': 'te_ds_llm_name',
    'Test - S1 S2 Type': 'te_s1s2',
    'Test - Significance Level': 'te_sig_lvl',
    'Test - Permutation Count': 'te_perm_cnt',
    'Test - Shuffled': 'te_shuffled',
    'Test - Sample Size': 'te_sample_size',
    'Test - Seed': 'te_seed',
    'Test - Test Size': 'test_size',
    'Test - SST Enabled': 'sst_enabled_te',
    'Test - SST Test Dataset': 'sst_test_dataset_te',
    'Test - SST Fill Dataset': 'sst_fill_dataset_te',
    'Test - SST Test Type': 'sst_test_type',
    'Test - SST Fill Type': 'sst_fill_type',
    'Test - SST True Data Ratio': 'sst_true_ratio_te',
    'Test - SST Strong Enabled': 'sst_strong_te',
    'Result - Test Power': 'test_power',
    'Result - Threshold Mean': 'threshold_mean',
    'Result - MMD Mean': 'mmd_mean'
}

VAR_NAME_TO_CSV_NAME = {v: k for k, v in CSV_NAME_TO_VAR_NAME.items()}
DATA_PATH = './test_logs/'
S1S2 = (('hh', 'mm'), ('hm'))
DEBUG = True


class Analysis(abc.ABC):
    def __init__(self, df, datasets_included: 'list[str]' = DATASETS):
        self.datasets_included = datasets_included
        self.df = df
        
    def aggregate_numpy_all_datasets(self, analysis_fun):
        '''Get analysis result for each database and a result for the aggregated data, combine the results horizontally.'''
        if len(self.datasets_included) < 1: raise ValueError("Database list must at least contain one dataset")
        
        # Get result for each database, and then concatenate together in axis 1.
        result = None
        for ds in self.datasets_included:
            result_ds = analysis_fun(self.df[self.df.te_ds_name == ds])
            result = np.concatenate((result, result_ds), axis=1) if result is not None else result_ds
            
        # Get result for the aggregated database
        temp = analysis_fun(self.df)
        result = np.concatenate((result, temp), axis=1)
        return result
    
    @abc.abstractmethod
    def get_numpy_single_dataset(self, df: pd.DataFrame) -> np.array:
        '''Analysis function that obtains the numpy result for a single dataset's data (df)'''
        pass
    
    def get_numpy(self) -> np.array:
        '''Return results in numpy array format'''
        return self.aggregate_numpy_all_datasets(self.get_numpy_single_dataset)
    
    @abc.abstractmethod
    def get_df(self) -> pd.DataFrame:
        '''Return results in DF format, which should build upon the numpy results.
        This should be the main function to call for analysis.'''
        pass
    
class Shuffle_Analysis(Analysis):
    def __init__(self, 
                 df: pd.DataFrame, 
                 datasets_included: 'list[str]' = DATASETS, 
                 shuffled_tr=(True, False), 
                 shuffled_te=(True, False)):
        super().__init__(df, datasets_included)
        self.shuffled_tr = shuffled_tr
        self.shuffled_te = shuffled_te
        self.name = "Analysis for whether training and testing data are shuffled"
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, 4))
        for i, (trs, tes) in enumerate(it.product(self.shuffled_tr, self.shuffled_te)):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[(df.tr_shuffled == trs) 
                    & (df.te_shuffled == tes) 
                    & (df.te_s1s2.map(lambda x: x in s1s2))
                    ].test_power
                result[2*j, i] = df_result.mean()
                result[2*j+1, i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
    
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Test Power Mean', 'Test Power Std. Dev.'))),
                    columns=pd.MultiIndex.from_product(
                        (
                            DATASETS+['Aggregated'], 
                            self.shuffled_tr, 
                            self.shuffled_te
                        ), 
                        names=('Database', 'Train Shuffled:', 'Test Shuffled:')
                    ))
        return df_result

class Linear_Size_Analysis(Analysis):
    def __init__(self, 
                 df: pd.DataFrame, 
                 datasets_included: 'list[str]' = DATASETS, 
                 lin_sizes: 'list[int]' = None):
        super().__init__(df, datasets_included)
        self.lin_sizes = self.df.linear_size.unique() if lin_sizes is None else lin_sizes
        self.name = "Analysis for linear layer size factor"
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, len(self.lin_sizes)))
        for i, ls in enumerate(self.lin_sizes):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[
                        (df.linear_size == ls) 
                        & (df.te_s1s2.map(lambda x: x in s1s2))
                    ].test_power
                result[2*j, i] = df_result.mean()
                result[2*j+1, i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
    
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Test Power Mean', 'Test Power Std. Dev.'))),
                    columns=pd.MultiIndex.from_product(
                        (
                            DATASETS+['Aggregated'], 
                            self.lin_sizes
                        ), 
                        names=('Database', 'Linear Size Multiple:')
                    ))
        return df_result

class Test_Sample_Size_Analysis(Analysis):
    def __init__(self, 
                 df: pd.DataFrame, 
                 datasets_included: 'list[str]' = DATASETS, 
                 sample_sizes: 'list[int]' = None):
        super().__init__(df, datasets_included)
        self.sample_sizes = self.df.te_sample_size.unique() if sample_sizes is None else sample_sizes
        self.name = "Analysis for sample size during testing"
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, len(self.sample_sizes)))
        for i, ss in enumerate(self.sample_sizes):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[
                        (df.te_sample_size == ss) 
                        & (df.te_s1s2.map(lambda x: x in s1s2))
                    ].test_power
                result[2*j, i] = df_result.mean()
                result[2*j+1, i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
    
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Test Power Mean', 'Test Power Std. Dev.'))),
                    columns=pd.MultiIndex.from_product(
                        (
                            DATASETS+['Aggregated'], 
                            self.sample_sizes
                        ), 
                        names=('Database', 'Test Sample Size:')
                    ))
        return df_result
    
class Permutation_Count_Analysis(Analysis):
    def __init__(self, 
                 df: pd.DataFrame, 
                 datasets_included: 'list[str]' = DATASETS, 
                 perm_cnts: 'list[int]' = None):
        super().__init__(df, datasets_included)
        self.perm_cnts = self.df.te_perm_cnt.unique() if perm_cnts is None else perm_cnts
        self.name = "Analysis for permutation count during testing"
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, len(self.perm_cnts)))
        for i, pc in enumerate(self.perm_cnts):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[
                        (df.te_perm_cnt == pc) 
                        & (df.te_s1s2.map(lambda x: x in s1s2))
                    ].test_power
                result[2*j, i] = df_result.mean()
                result[2*j+1, i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
    
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Test Power Mean', 'Test Power Std. Dev.'))),
                    columns=pd.MultiIndex.from_product(
                        (
                            DATASETS+['Aggregated'], 
                            self.perm_cnts
                        ), 
                        names=('Database', 'Test Permutation Count:')
                    ))
        return df_result
    
class LLM_Analysis(Analysis):
    def __init__(self, 
                 df: pd.DataFrame, 
                 datasets_included: 'list[str]' = DATASETS, 
                 llms: 'list[str]' = None):
        super().__init__(df, datasets_included)
        self.llms = self.df.te_ds_llm_name.unique() if llms is None else llms
        self.name = "Analysis for different LLMs"
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, len(self.llms)))
        for i, llm in enumerate(self.llms):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[
                        (df.te_ds_llm_name == llm) 
                        & (df.te_s1s2.map(lambda x: x in s1s2))
                    ].test_power
                result[2*j, i] = df_result.mean()
                result[2*j+1, i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
    
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Test Power Mean', 'Test Power Std. Dev.'))),
                    columns=pd.MultiIndex.from_product(
                        (
                            DATASETS+['Aggregated'], 
                            self.llms
                        ), 
                        names=('Database', 'Test LLM:')
                    ))
        return df_result
    
class SST_True_Data_Ratio_Analysis(Analysis):
    def __init__(self, 
                 df: pd.DataFrame, 
                 datasets_included: 'list[str]' = DATASETS, 
                 true_data_ratio: 'list[int]' = None):
        super().__init__(df, datasets_included)
        self.true_data_ratios = self.df.sst_true_ratio_te.unique() if true_data_ratio is None else true_data_ratio
        self.true_data_ratios.sort()
        self.name = "Analysis for single sample test true data ratio"
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, len(self.true_data_ratios)))
        for i, tdr in enumerate(self.true_data_ratios):
            for j, s1s2 in enumerate(S1S2):
                df_s1s2 = df.apply(lambda row: row['sst_test_type'][0] + row['sst_fill_type'][0], axis=1)
                df_result = df[
                        (df.sst_true_ratio_te == tdr) 
                        & (df_s1s2.map(lambda x: x in s1s2))
                    ].test_power
                result[2*j, i] = df_result.mean()
                result[2*j+1, i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
    
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Test Power Mean', 'Test Power Std. Dev.'))),
                    columns=pd.MultiIndex.from_product(
                        (
                            DATASETS+['Aggregated'], 
                            self.true_data_ratios
                        ), 
                        names=('Database', 'True data ratio:')
                    ))
        return df_result
    
class SST_Dataset_Analysis(Analysis):
    def __init__(self, 
                 df: pd.DataFrame, 
                 test_datasets: 'list[str]' = None,
                 fill_datasets: 'list[str]' = None):
        super().__init__(df, DATASETS)
        self.test_datasets = self.df.sst_test_dataset_te.unique() if test_datasets is None else test_datasets
        self.fill_datasets = self.df.sst_fill_dataset_te.unique() if fill_datasets is None else fill_datasets
        self.name = "Analysis for single sample test with different combination of test and fill datasets"
    
    def get_numpy_single_dataset(self, df: pd.DataFrame) -> np.array:
        return self.get_numpy(df)
    
    def get_numpy(self, df: pd.DataFrame):
        result = np.zeros((4, len(self.test_datasets) * len(self.fill_datasets)))
        for i, tds in enumerate(self.test_datasets):
            for j, fds in enumerate(self.fill_datasets):
                for k, s1s2 in enumerate(S1S2):
                    df_s1s2 = df.apply(lambda row: row['sst_test_type'][0] + row['sst_fill_type'][0], axis=1)
                    df_result = df[
                            (df.sst_test_dataset_te == tds) 
                            & (df.sst_fill_dataset_te == fds)
                            & (df_s1s2.map(lambda x: x in s1s2))
                        ].test_power
                    result[2*k, len(self.fill_datasets)*i+j] = df_result.mean()
                    result[2*k+1, len(self.fill_datasets)*i+j] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy(self.df)
    
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Test Power Mean', 'Test Power Std. Dev.'))),
                    columns=pd.MultiIndex.from_product(
                        (
                            self.test_datasets,
                            self.fill_datasets
                        ), 
                        names=('Test Dataset', 'Fill Dataset')
                    ))
        return df_result
            
###########################################
# Analysis access points
###########################################

def param_acc_analysis(df, logger):
    logger.info("Starting analysis...")
    
    gsa_df = Shuffle_Analysis(df, datasets_included=DATASETS).get_df().round(4)
    logger.info(
        "\nResult of Shuffled Data Analysis:\n"
        f"{gsa_df}\n")
    
    glsa_df = Linear_Size_Analysis(df, datasets_included=DATASETS).get_df().round(4)
    logger.info(
        "\nResult of Linear Layer Size Analysis:\n"
        f"{glsa_df}\n")
    
    gtbsa_df = Test_Sample_Size_Analysis(df, datasets_included=DATASETS).get_df().round(4)
    logger.info(
        "\nResult of Test Batch Size Analysis:\n"
        f"{gtbsa_df}\n")
    
    gpca_df = Permutation_Count_Analysis(df, datasets_included=DATASETS).get_df().round(4)
    logger.info(
        "\nResult of Test Permutation Count Analysis:\n"
        f"{gpca_df}\n")
    
    return gsa_df, glsa_df, gtbsa_df, gpca_df
    
    
def param_acc_filtered_analysis(df, logger):
    logger.info("Starting analysis...")
    
    gtbsa_df = Test_Sample_Size_Analysis(df, datasets_included=DATASETS).get_df().round(4)
    logger.info(
        "\nResult of Test Batch Size Analysis:\n"
        f"{gtbsa_df}\n")
    
    gpca_df = Permutation_Count_Analysis(df, datasets_included=DATASETS).get_df().round(4)
    logger.info(
        "\nResult of Test Permutation Count Analysis:\n"
        f"{gpca_df}\n")
    
    return gtbsa_df, gpca_df

def database_best_acc_analysis(df, logger):
    '''Using best paramter values and only test the batch size and permutation count parameters'''
    logger.info("Starting analysis...")
    
    # Filter data to choose the best param
    df_filterd = df[
        (df.tr_shuffled == False)
        & (df.te_shuffled == True)
        & (df.linear_size == 3)
    ]

    gsa_df = Shuffle_Analysis(df_filterd, datasets_included=DATASETS).get_df().round(4)
    logger.info(
        "\nResult of Best Model Shuffled Data Analysis:\n"
        f"{gsa_df}\n")
    
    glsa_df = Linear_Size_Analysis(df_filterd, datasets_included=DATASETS).get_df().round(4)
    logger.info(
        "\nResult of Best Model Linear Layer Size Analysis:\n"
        f"{glsa_df}\n")
    
    gtbsa_df = Test_Sample_Size_Analysis(df_filterd, datasets_included=DATASETS).get_df().round(4)
    logger.info(
        "\nResult of Best Model Test Batch Size Analysis:\n"
        f"{gtbsa_df}\n")
    
    gpca_df = Permutation_Count_Analysis(df_filterd, datasets_included=DATASETS).get_df().round(4)
    logger.info(
        "\nResult of Best Model Test Permutation Count Analysis:\n"
        f"{gpca_df}\n")
    
    return gsa_df, glsa_df, gtbsa_df, gpca_df

def llm_acc_analysis(df, logger):
    logger.info("Starting analysis...")
    
    # Filter data to choose the best param
    df_filterd = df[
        (df.te_sample_size == 10)
        & (df.linear_size == 3)
    ]
    
    llm_df = LLM_Analysis(df_filterd, datasets_included=DATASETS).get_df().round(4)
    logger.info(
        "\nResult of LLM Analysis:\n"
        f"{llm_df}\n")

    return llm_df

def true_data_ratio_acc_analysis(df, logger):
    logger.info("Starting analysis...")
    
    # Filter data to choose the best param
    df_filterd = df[
        (df.te_sample_size == 10)
        & (df.linear_size == 3)
    ]
    
    tdr_df = SST_True_Data_Ratio_Analysis(df_filterd, datasets_included=DATASETS).get_df().round(4)
    logger.info(
        f"\nResult of SST_True_Data_Ratio_Analysis:\n"
        f"{tdr_df}\n")

    return tdr_df

def sst_database_acc_analysis(df, logger):
    logger.info("Starting analysis...")
    
    # Filter data to choose the best param
    df_filterd = df[
        (df.te_sample_size == 10)
        & (df.linear_size == 3)
    ]
    
    sda_df = SST_Dataset_Analysis(df_filterd).get_df().round(4)
    logger.info(
        f"\nResult of SST_Dataset_Analysis:\n"
        f"{sda_df}\n")

    return sda_df


# def param_proportion_acc_analysis():
#     '''Accuracy analysis that instead focus on proporation of a certain paramter value over a specific accuracy threshold'''
#     start_time_str = util.get_current_time_str()
#     test_file_name = 'test_20230817055948.csv'
    
#     logger = util.setup_logs(
#         file_path=f"./analysis_logs/analysis_{start_time_str}.log",
#         id=start_time_str,
#     )
    
#     # Get data
#     logger.info(f"Getting and pre-processing data from file {test_file_name}")
#     df = get_data(test_file_name)
#     df = get_preprocessed_df(df)
    
#     # Analysis from existing data
#     acc_threshold = 0.85
#     logger.info(f"Starting analysis for type 2 accuracy above {acc_threshold}")
#     results = get_proportion_analysis_for_acc_above(df, acc_threshold=acc_threshold)
    
#     # Log data
#     for r in results.values():
#         logger.info(f"\n{r}\n")
  
def main():
    start_time_str = util.get_current_time_str()
    data_file = 'test_20231002063213 sst dataset.csv'
    log_dir = './analysis_logs/'
    analysis_name = f"analysis_{start_time_str}"
    
    # Setup logs
    
    logger = util.setup_logs(
        file_path=os.path.join(log_dir, analysis_name + '.log'),
        id=start_time_str,
        is_debug=DEBUG
    )
    
    logger.info(f"Using file: {data_file}")
    
    # Data read and preprocess 
    df = pd.read_csv(os.path.join(DATA_PATH, data_file))
    df = df.drop(columns=df.columns[0])
    df = df.rename(columns=CSV_NAME_TO_VAR_NAME)
    
    # Start
    # gsa_df, glsa_df, gtbsa_df, gpca_df = param_acc_analysis(df, logger)
    # database_best_acc_analysis(df, logger)
    # llm_df = llm_acc_analysis(df, logger)
    # tdr_df = true_data_ratio_acc_analysis(df, logger)
    sda_df = sst_database_acc_analysis(df, logger)
    
    # Save to csv
    csv_path = os.path.join(log_dir, analysis_name + '.csv')
    sda_df.to_csv(csv_path)
    logger.info(f'Result saved to csv at {csv_path}')

if __name__ == '__main__':
    main()