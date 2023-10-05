import os
import itertools as it
import abc
from argparse import ArgumentParser

import pandas as pd
import numpy as np

from external.dataset_loader import DATASETS
import util

DATA_PATH = './test_logs/'
S1S2 = (('hh', 'mm'), ('hm', 'mh'))
DEBUG = True


class Analysis(abc.ABC):
    def __init__(self, df, datasets_included: 'list[str]' = [[d] for d in DATASETS]):
        self.datasets_included = datasets_included
        self.df = df
        
    def aggregate_numpy_all_datasets(self, analysis_fun):
        '''Get analysis result for each database and a result for the aggregated data, combine the results horizontally.'''
        if len(self.datasets_included) < 1: raise ValueError("Database list must at least contain one dataset")
        
        # Get result for each database, and then concatenate together in axis 1.
        result = None
        for ds in self.datasets_included:
            result_ds = analysis_fun(self.df[self.df.te_tst_datasets == str(ds)])
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
    
    def get_postprocessed_df(self, logger=None) -> pd.DataFrame:
        '''Return post-processed DF results, and log the results'''
        result_df = self.get_df()
        result_df = result_df.round(4)
        if logger is not None:
            logger.info(
                f'Result of {self.description}:\n'
                f'{result_df}\n'
            )
        return result_df

class Seed_Analysis(Analysis):
    description = 'Analysis for a model configuration different training seeds'
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 datasets_included: 'list[str]' = [[d] for d in DATASETS], 
                 seeds: 'list[int]' = None,
                 ):
        super().__init__(df, datasets_included)
        self.seeds = self.df.tr_seed.unique() if seeds is None else seeds
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((12, 1))
        for i, seed in enumerate(self.seeds):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[(df.tr_seed == seed) 
                    & (df.te_tst_s1s2_type.map(lambda x: x in s1s2))
                    ].test_power
                result[4*i + 2*j, 0] = df_result.mean()
                result[4*i + 2*j+1, 0] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (
                            self.seeds,
                            ('Type 1', 'Type 2'), 
                            ('Mean', 'Std.')
                        ),
                        names=(
                            'Seeds',
                            'Test Type',
                            'Test Power'
                        )),
                    columns=pd.Index(
                        data=DATASETS+['Aggregated'],  
                        name=('Database')
                    ))
        return df_result
   
class Shuffle_Analysis(Analysis):
    description = 'Analysis for whether training and testing data are shuffled'
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 datasets_included: 'list[str]' = [[d] for d in DATASETS], 
                 shuffled_tr=(True, False), 
                 shuffled_te=(True, False)):
        super().__init__(df, datasets_included)
        self.shuffled_tr = shuffled_tr
        self.shuffled_te = shuffled_te
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, 4))
        for i, (trs, tes) in enumerate(it.product(self.shuffled_tr, self.shuffled_te)):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[(df.tr_shuffle == trs) 
                    & (df.te_shuffle == tes) 
                    & (df.te_tst_s1s2_type.map(lambda x: x in s1s2))
                    ].test_power
                result[2*j, i] = df_result.mean()
                result[2*j+1, i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Mean', 'Std.'))),
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
    description = 'Analysis for linear layer size factor'
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 datasets_included: 'list[str]' = [[d] for d in DATASETS], 
                 lin_sizes: 'list[int]' = None):
        super().__init__(df, datasets_included)
        self.lin_sizes = self.df.tr_lin_size.unique() if lin_sizes is None else lin_sizes
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, len(self.lin_sizes)))
        for i, ls in enumerate(self.lin_sizes):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[
                        (df.tr_lin_size == ls) 
                        & (df.te_tst_s1s2_type.map(lambda x: x in s1s2))
                    ].test_power
                result[2*j, i] = df_result.mean()
                result[2*j+1, i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Mean', 'Std.'))),
                    columns=pd.MultiIndex.from_product(
                        (
                            DATASETS+['Aggregated'], 
                            self.lin_sizes
                        ), 
                        names=('Database', 'Linear Size Multiple:')
                    ))
        return df_result

class Test_Sample_Size_Analysis(Analysis):
    description = 'Analysis for sample size during testing'
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 datasets_included: 'list[str]' = [[d] for d in DATASETS], 
                 sample_sizes: 'list[int]' = None):
        super().__init__(df, datasets_included)
        self.sample_sizes = self.df.te_sample_size.unique() if sample_sizes is None else sample_sizes
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, len(self.sample_sizes)))
        for i, ss in enumerate(self.sample_sizes):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[
                        (df.te_sample_size == ss) 
                        & (df.te_tst_s1s2_type.map(lambda x: x in s1s2))
                    ].test_power
                result[2*j, i] = df_result.mean()
                result[2*j+1, i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Mean', 'Std.'))),
                    columns=pd.MultiIndex.from_product(
                        (
                            DATASETS+['Aggregated'], 
                            self.sample_sizes
                        ), 
                        names=('Database', 'Test Sample Size:')
                    ))
        return df_result
    
class Permutation_Count_Analysis(Analysis):
    description = 'Analysis for permutation count during testing'
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 datasets_included: 'list[str]' = [[d] for d in DATASETS], 
                 perm_cnts: 'list[int]' = None):
        super().__init__(df, datasets_included)
        self.perm_cnts = self.df.te_perm_cnt.unique() if perm_cnts is None else perm_cnts
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, len(self.perm_cnts)))
        for i, pc in enumerate(self.perm_cnts):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[
                        (df.te_perm_cnt == pc) 
                        & (df.te_tst_s1s2_type.map(lambda x: x in s1s2))
                    ].test_power
                result[2*j, i] = df_result.mean()
                result[2*j+1, i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Mean', 'Std.'))),
                    columns=pd.MultiIndex.from_product(
                        (
                            DATASETS+['Aggregated'], 
                            self.perm_cnts
                        ), 
                        names=('Database', 'Test Permutation Count:')
                    ))
        return df_result
    
class LLM_Analysis(Analysis):
    description = 'Analysis for different LLMs'

    def __init__(self, 
                 df: pd.DataFrame, 
                 datasets_included: 'list[str]' = [[d] for d in DATASETS], 
                 llms: 'list[str]' = None):
        super().__init__(df, datasets_included)
        self.llms = self.df.te_data_llm.unique() if llms is None else llms
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, len(self.llms)))
        for i, llm in enumerate(self.llms):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[
                        (df.te_data_llm == llm) 
                        & (df.te_tst_s1s2_type.map(lambda x: x in s1s2))
                    ].test_power
                result[2*j, i] = df_result.mean()
                result[2*j+1, i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Mean', 'Std.'))),
                    columns=pd.MultiIndex.from_product(
                        (
                            DATASETS+['Aggregated'], 
                            self.llms
                        ), 
                        names=('Database', 'Test LLM:')
                    ))
        return df_result
    
class SST_True_Data_Ratio_Analysis(Analysis):
    description = 'Analysis for single sample test true data ratio'

    def __init__(self, 
                 df: pd.DataFrame, 
                 datasets_included: 'list[str]' = [[d] for d in DATASETS], 
                 true_data_ratio: 'list[int]' = None):
        super().__init__(df, datasets_included)
        self.true_data_ratios = self.df.te_sst_true_ratio.unique() if true_data_ratio is None else true_data_ratio
        self.true_data_ratios.sort()
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, len(self.true_data_ratios)))
        for i, tdr in enumerate(self.true_data_ratios):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[
                        (df.te_sst_true_ratio == tdr) 
                        & (df.sst_userfill_type.map(lambda x: x in s1s2))
                    ].test_power
                result[2*j, i] = df_result.mean()
                result[2*j+1, i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Mean', 'Std.'))),
                    columns=pd.MultiIndex.from_product(
                        (
                            DATASETS+['Aggregated'], 
                            self.true_data_ratios
                        ), 
                        names=('Database', 'True data ratio:')
                    ))
        return df_result
    
class SST_Dataset_Analysis(Analysis):
    description = 'Analysis for single sample test with different combination of test and fill datasets'
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 test_datasets: 'list[str]' = None,
                 fill_datasets: 'list[str]' = None):
        super().__init__(df, DATASETS)
        self.test_datasets = self.df.te_sst_user_dataset.unique() if test_datasets is None else test_datasets
        self.fill_datasets = self.df.te_sst_fill_dataset.unique() if fill_datasets is None else fill_datasets
    
    def get_numpy_single_dataset(self, df: pd.DataFrame) -> np.array:
        return self.get_numpy(df)
    
    def get_numpy(self, df: pd.DataFrame):
        result = np.zeros((4, len(self.test_datasets) * len(self.fill_datasets)))
        for i, tds in enumerate(self.test_datasets):
            for j, fds in enumerate(self.fill_datasets):
                for k, s1s2 in enumerate(S1S2):
                    df_result = df[
                            (df.te_sst_user_dataset == tds) 
                            & (df.te_sst_fill_dataset == fds)
                            & (df.sst_userfill_type.map(lambda x: x in s1s2))
                        ].test_power
                    result[2*k, len(self.fill_datasets)*i+j] = df_result.mean()
                    result[2*k+1, len(self.fill_datasets)*i+j] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy(self.df)
    
        df_result = pd.DataFrame(result,
                    index=pd.MultiIndex.from_product(
                        (('Type 1', 'Type 2'), 
                        ('Mean', 'Std.'))),
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


# def param_proportion_acc_analysis():
#     '''Accuracy analysis that instead focus on proporation of a certain paramter value over a specific accuracy threshold'''
#     start_time_str = util.get_current_time_str()
#     tefile_name = 'test_20230817055948.csv'
    
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
    data_file = 'test_20231005173107_SQuAD1_sample_size.csv'
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
    
    # Filtering
    # df = df[
    #     (df.tr_shuffle == False)
    #     & (df.te_shuffle == True)
    #     & (df.tr_lin_size == 3)
    # ]
    
    # Start

    result = Test_Sample_Size_Analysis(df).get_postprocessed_df(logger)
    # seeds_df = Seed_Analysis(df).get_df().round(4)
    
    # Save to csv
    csv_path = os.path.join(log_dir, analysis_name + '.csv')
    result.to_csv(csv_path)
    logger.info(f'Result saved to csv at {csv_path}')

if __name__ == '__main__':
    main()