import abc
import itertools as it
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simple_parsing import ArgumentParser

from external.dataset_loader import DATASETS
import util

###########################################
# Tabular Analysis
###########################################

class TabularAnalysis(abc.ABC):
    description = None
    
    def __init__(self, df):
        df = df.drop(columns=df.columns[0])
        self.df = df
        
        # Shared row indices
        self.row_indices = pd.MultiIndex.from_product(
            iterables = (
                ['Aggregated']+DATASETS,
                ('Type 1', 'Type 2'), 
                ('Mean', 'Std.')
            ),
            names = ('Database', 'Test Type', 'Rejection Rate')
        )
        self.row_indicies_without_std = pd.MultiIndex.from_product(
            iterables = (
                ['Aggregated']+DATASETS,
                ('Type 1', 'Type 2'), 
            ),
            names = ('Database', 'Test Type')
        )
        
    def aggregate_numpy_all_datasets(self, analysis_fun):
        '''Get analysis result for each database and a result for the aggregated data, combine the results vertically.'''
        # Get result for each database, and then concatenate together in axis 1.
        result = None
        for ds in self.df.te_tst_datasets.unique():
            result_ds = analysis_fun(self.df[self.df.te_tst_datasets == ds])
            result = np.concatenate((result, result_ds), axis=0) if result is not None else result_ds
            
        # Get result for the aggregated database
        aggregated_result = analysis_fun(self.df)
        result = np.concatenate((aggregated_result, result), axis=0)
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

class TabularAnalysisSeed(TabularAnalysis):
    description = 'Analysis for a model configuration different training seeds'
    
    def __init__(self, 
                 df: pd.DataFrame,
                 seeds: 'list[int]' = None,
                 ):
        super().__init__(df)
        self.seeds = self.df.tr_seed.unique() if seeds is None else seeds
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, 3))
        for i, seed in enumerate(self.seeds):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[(df.tr_seed == seed) 
                    & (df.te_tst_s1s2_type.map(lambda x: x in s1s2))
                    ].test_power
                result[2*j, i] = df_result.mean()
                result[2*j+1, i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
        df_result = pd.DataFrame(result,
                    index = self.row_indices,
                    columns = pd.Index(
                        data = self.seeds,  
                        name = 'Seeds'
                    ))
        return df_result
   
class TabularAnalysisShuffle(TabularAnalysis):
    description = 'Analysis for whether training and testing data are shuffled'
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 shuffled_tr=(True, False), 
                 shuffled_te=(True, False)):
        super().__init__(df)
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
                    index = self.row_indices,
                    columns = pd.MultiIndex.from_product(
                        (
                            self.shuffled_tr, 
                            self.shuffled_te
                        )
                    ))
        return df_result

class TabularAnalysisSize(TabularAnalysis):
    description = 'Analysis for linear layer size factor'
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 lin_sizes: 'list[int]' = None):
        super().__init__(df)
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
                    index = self.row_indices,
                    columns = pd.Index(
                        data = self.lin_sizes,
                        name = 'Linear Size Multiple',
                    ))
        return df_result

class TabularAnalysisSampleSize(TabularAnalysis):
    description = 'Analysis for sample size during testing'
    
    def __init__(self, 
                 df: pd.DataFrame,
                 sample_sizes: 'list[int]' = None):
        super().__init__(df)
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
                    index = self.row_indices,
                    columns = pd.Index(
                        data = self.sample_sizes,
                        name = 'Test Sample Size',
                    ))
        return df_result
    
    
class TabularAnalysisSampleSize(TabularAnalysis):
    description = 'Analysis for sample size during testing'
    
    def __init__(self, 
                 df: pd.DataFrame,
                 sample_sizes: 'list[int]' = None):
        super().__init__(df)
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
                    index = self.row_indices,
                    columns = pd.Index(
                        data = self.sample_sizes,
                        name = 'Test Sample Size',
                    ))
        return df_result
    
class TabularAnalysisLLMSampleSize(TabularAnalysis):
    description = 'Analysis for sample size during testing'
    
    def __init__(self, 
                 df: pd.DataFrame,
                 sample_sizes: 'list[int]' = None):
        super().__init__(df)
        self.sample_sizes = self.df.te_sample_size.unique() if sample_sizes is None else sample_sizes
        self.llms = self.df.te_data_llm.unique()
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((4, len(self.sample_sizes)*len(self.llms)))
        for i, ss in enumerate(self.sample_sizes):
            for j, llm in enumerate(self.llms):
                for k, s1s2 in enumerate(S1S2):
                    df_result = df[
                            (df.te_sample_size == ss) 
                            & (df.te_data_llm == llm)
                            & (df.te_tst_s1s2_type.map(lambda x: x in s1s2))
                        ].test_power
                    result[2*k, j*len(self.sample_sizes)+i] = df_result.mean()
                    result[2*k+1, j*len(self.sample_sizes)+i] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy()
        df_result = pd.DataFrame(result,
                    index = self.row_indices,
                    columns = pd.MultiIndex.from_product(
                        iterables = (self.llms, self.sample_sizes),
                        names = ('LLM', 'Test Sample Size'),
                    ))
        return df_result
    
    
class TabularAnalysisPermutationCount(TabularAnalysis):
    description = 'Analysis for permutation count during testing'
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 perm_cnts: 'list[int]' = None):
        super().__init__(df)
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
                    index = self.row_indices,
                    columns = pd.Index(
                        data = self.perm_cnts,
                        name = 'Test Permutation Count',
                    ))
        return df_result
    
class TabularAnalysisLLM(TabularAnalysis):
    description = 'Analysis for different LLMs'

    def __init__(self, 
                 df: pd.DataFrame, 
                 llms: 'list[str]' = None):
        super().__init__(df)
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
                    index = self.row_indices,
                    columns = pd.Index(
                        data = self.llms,
                        name = 'Test LLM:',
                    ))
        return df_result
    
class TabularAnalysisSSTTrueDataRatio(TabularAnalysis):
    '''Requirements: SST user dataset is the same as SST fill dataset'''
    description = 'Analysis for single sample test true data ratio'

    def __init__(self, 
                 df: pd.DataFrame, 
                 true_data_ratio: 'list[int]' = None):
        super().__init__(df)
        self.true_data_ratios = self.df.te_sst_true_ratio.unique() if true_data_ratio is None else true_data_ratio
        self.true_data_ratios.sort()
        self.s1s2s = self.df.te_sst_userfill_type.unique()
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((2, len(self.true_data_ratios)))
        for i, tdr in enumerate(self.true_data_ratios):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[
                        (df.te_sst_true_ratio == tdr) 
                        & (df.te_sst_userfill_type.map(lambda x: x in s1s2))
                    ].test_power
                result[j, i] = df_result.mean()
        return result
    
    def aggregate_numpy_all_datasets(self, analysis_fun):
        '''Get analysis result for each database and a result for the aggregated data, combine the results vertically.'''
        # Get result for each database, and then concatenate together in axis 1.
        result = None
        for ds in self.df.te_sst_user_dataset.unique():
            result_ds = analysis_fun(self.df[self.df.te_sst_user_dataset == ds])
            result = np.concatenate((result, result_ds), axis=0) if result is not None else result_ds
            
        # Get result for the aggregated database
        aggregated_result = analysis_fun(self.df)
        result = np.concatenate((aggregated_result, result), axis=0)
        return result
    
    def get_df(self):
        result = self.get_numpy()
        df_result = pd.DataFrame(result,
                    index = self.row_indicies_without_std,
                    columns = pd.Index(
                        data = self.true_data_ratios,
                        name = 'True data ratio',
                    ))
        return df_result
    
class TabularAnalysisSSTDataset(TabularAnalysis):
    description = 'Analysis for single sample test with different combination of test and fill datasets'
    
    def __init__(self, 
                 df: pd.DataFrame,):
        super().__init__(df)
        self.user_datasets = self.df.te_sst_user_dataset.unique()
        self.fill_datasets = self.df.te_sst_fill_dataset.unique()
    
    def get_numpy_single_dataset(self, df: pd.DataFrame) -> np.array:
        return self.get_numpy(df)
    
    def get_numpy(self, df: pd.DataFrame):
        result = np.zeros((4 * len(self.user_datasets), len(self.fill_datasets)))
        for i, tds in enumerate(self.user_datasets):
            for j, fds in enumerate(self.fill_datasets):
                for k, s1s2 in enumerate(S1S2):
                    df_result = df[
                            (df.te_sst_user_dataset == tds) 
                            & (df.te_sst_fill_dataset == fds)
                            & (df.te_sst_userfill_type.map(lambda x: x in s1s2))
                        ].test_power
                    result[i*4 + 2*k, j] = df_result.mean()
                    result[i*4 + 2*k+1, j] = df_result.std()
        return result
    
    def get_df(self):
        result = self.get_numpy(self.df)
    
        df_result = pd.DataFrame(result,
                    index = pd.MultiIndex.from_product(
                        iterables=(
                            self.user_datasets,
                            ('Type 1', 'Type 2'), 
                            ('Mean', 'Std.')
                        ),
                        names=('User Dataset', 'Test Type', 'Rejection Rate')
                    ),
                    columns = pd.Index(
                        data=self.fill_datasets,
                        name='Fill Dataset',
                    ))
        return df_result

class TabularAnalysisSSTOptimalLLM(TabularAnalysis):
    '''Requirements: Single true data ratio, SST user dataset is the same as SST fill dataset'''
    description = 'Analysis for single sample test on different LLMs with an optimal (single) True data ratio'

    def __init__(self, 
                 df: pd.DataFrame): 
        super().__init__(df)
        assert len(self.df.te_sst_true_ratio) != 1, "This analysis only works for results with a single true data ratio"
        self.LLMs = self.df.te_data_llm.unique()
        self.s1s2s = self.df.te_sst_userfill_type.unique()
        
    def get_numpy_single_dataset(self, df: pd.DataFrame):
        result = np.zeros((2, len(self.LLMs)))
        for i, llm in enumerate(self.LLMs):
            for j, s1s2 in enumerate(S1S2):
                df_result = df[
                        (df.te_data_llm == llm) 
                        & (df.te_sst_userfill_type.map(lambda x: x in s1s2))
                    ].test_power
                result[j, i] = df_result.mean()
        return result
    
    def aggregate_numpy_all_datasets(self, analysis_fun):
        '''Get analysis result for each database and a result for the aggregated data, combine the results vertically.'''
        # Get result for each database, and then concatenate together in axis 1.
        result = None
        for ds in self.df.te_sst_user_dataset.unique():
            result_ds = analysis_fun(self.df[self.df.te_sst_user_dataset == ds])
            result = np.concatenate((result, result_ds), axis=0) if result is not None else result_ds
            
        # Get result for the aggregated database
        aggregated_result = analysis_fun(self.df)
        result = np.concatenate((aggregated_result, result), axis=0)
        return result
    
    def get_df(self):
        result = self.get_numpy()
        df_result = pd.DataFrame(result,
                    index = self.row_indicies_without_std,
                    columns = pd.Index(
                        data = self.LLMs,
                        name = 'LLM',
                    ))
        return df_result

###########################################
# Graphic Analysis
###########################################

class GraphAnalysis(abc.ABC):
    '''Abstract class for graph analysis'''
    description = None
    
    def __init__(self, df) -> None:
        self.df = df
    
    def add_desired_line(g, label_x, test_type):
        target = 0.05 if test_type == 1 else 1.0
        def specs(x, **kwargs):
            plt.axhline(target, c='red', ls='--', lw=1.5)
        g.map(specs, 'test_power')  
        g.axes.flat[0].text(label_x, 0.994 if test_type == 2 else 0.049, "Target", color="red")

    @abc.abstractmethod
    def plot_graph(self):
        pass
    
class GraphAnalysisShuffle(GraphAnalysis):
    description = 'Analysis for whether training and testing data are shuffled'
    
    def __init__(self, df) -> None:
        super().__init__(df)
        
    def plot_graph(self):
        for test_type in [1, 2]:
            df = self.df
            
            # Type
            df = df[df['te_tst_s1s2_type'].map(lambda x: x in (['hm', 'mh'] if test_type == 2 else ['hh', 'mm']))]

            # Add required rows
            for tr in [True, False]:
                for te in [True, False]:
                    mean = df[(df['tr_shuffle'] == tr) & (df['te_shuffle'] == te)]['test_power'].mean()
                    row = {'tr_shuffle': tr, 'te_shuffle': te, 'te_tst_datasets': 'Average', 'test_power': mean}
                    row = pd.DataFrame(columns=df.columns, data=[row])
                    df = pd.concat([df, row], axis=0)
                    
            # Add required colums
            df['trte_shuffle'] = df.apply(lambda row: f'{row.tr_shuffle} {row.te_shuffle}', axis=1)
            df.sort_values(by=['trte_shuffle'], inplace=True)

            # Plot
            sns.set_style("whitegrid")
            g = sns.catplot(df, kind='bar', x='trte_shuffle', y='test_power', hue='te_tst_datasets', sharey=True, aspect=2, capsize=0.2)
            for ax in g.axes.flat:
                for c in ax.containers:
                    labels = [f'{v.get_height():.3f}' for v in c]
                    ax.bar_label(c, labels=labels, label_type='center', color='white', fontsize=9, fontweight='bold')

            # Add desired line
            GraphAnalysis.add_desired_line(g, 3.5, test_type)

            # Style
            g.despine(left=True)
            g.set_xlabels('(Train data shuffled, Test data shuffled)')
            g.set_ylabels(f'Type-{"I" if test_type == 1 else "II"} Rejection Rate (Mean)')
            g.legend.set_title('Test Dataset')
            plt.show()           

class GraphAnalysisLinearSize(GraphAnalysis):
    description = 'Analysis for linear layer size multiple'
    
    def __init__(self, df) -> None:
        super().__init__(df)
        
    def plot_graph(self):
        for test_type in [1, 2]:
            df = self.df
        
            # Type
            df = df[df['te_tst_s1s2_type'].map(lambda x: x in (['hm', 'mh'] if test_type == 2 else ['hh', 'mm']))]

            # Add required rows
            for ls in df.tr_lin_size.unique():
                mean = df[df.tr_lin_size == ls]['test_power'].mean()
                row = {'tr_lin_size': ls, 'te_tst_datasets': 'Average', 'test_power': mean}
                row = pd.DataFrame(columns=df.columns, data=[row])
                df = pd.concat([df, row], axis=0)

            # Plot
            sns.set_style("whitegrid")
            g = sns.catplot(df, kind='bar', x='tr_lin_size', y='test_power', hue='te_tst_datasets', sharey=True, aspect=1.5, capsize=0.2, err_kws={'linewidth': 1})
            for ax in g.axes.flat:
                for c in ax.containers:
                    labels = [f'{v.get_height():.3f}' for v in c]
                    ax.bar_label(c, labels=labels, label_type='center', color='white', fontsize=9, fontweight='bold')

            # Add desired line
            GraphAnalysis.add_desired_line(g, 1.5, test_type)

            # Style
            g.despine(left=True)
            g.set_xlabels('Model Linear Layer Size Multiple')
            g.set_ylabels(f'Type-{"I" if test_type == 1 else "II"} Rejection Rate (Mean)')
            g.legend.set_title('Test Dataset')
            plt.show()    

class GraphAnalysisPermCnt(GraphAnalysis):
    description = 'Analysis for permutation count during testing'
    
    def __init__(self, df) -> None:
        super().__init__(df)
        
    def plot_graph(self):
        for test_type in [1, 2]:
            df = self.df
            
            # Type
            df = df[df['te_tst_s1s2_type'].map(lambda x: x in (['hm', 'mh'] if test_type == 2 else ['hh', 'mm']))]

            # Add required rows
            for x in df.te_perm_cnt.unique():
                mean = df[df.te_perm_cnt == x]['test_power'].mean()
                row = {'te_perm_cnt': x, 'te_tst_datasets': 'Average', 'test_power': mean}
                row = pd.DataFrame(columns=df.columns, data=[row])
                df = pd.concat([df, row], axis=0)

            # Plot
            sns.set_style("whitegrid")
            g = sns.catplot(df, kind='point', x='te_perm_cnt', y='test_power', hue='te_tst_datasets', sharey=True, aspect=1.5, capsize=0.2, err_kws={'linewidth': 1})

            # Add desired line
            GraphAnalysis.add_desired_line(g, 4.5, test_type)

            # Style
            g.despine(left=True)
            g.set_xlabels('Two-Sample Test Permutation Count')
            g.set_ylabels(f'Type-{"I" if test_type == 1 else "II"} Rejection Rate (Mean)')
            g.legend.set_title('Test Dataset')
            plt.show()         

class GraphAnalysisPermCntTiming(GraphAnalysis):
    description = 'Analysis for test time for different permutation counts, result is hard-coded'
    
    def __init__(self, df) -> None:
        super().__init__(df)
        
    def plot_graph(self):
        df = pd.DataFrame({
            'te_perm_cnt': [50, 100, 200, 300, 400],
            'time': [2.9, 3.2, 4.9, 5.4, 6.9]
        })
        sns.set_style("whitegrid")
        g = sns.catplot(df, kind='point', x='te_perm_cnt', y='time', sharey=True, aspect=1.3, capsize=0.2, err_kws={'linewidth': 1})
        
        # Style
        g.despine(left=True)
        g.set_xlabels('Two-Sample Test Permutation Count')
        g.set_ylabels(f'Test Running Time (s)')
        plt.show() 

class GraphAnalysisSampleSize(GraphAnalysis):
    description = 'Analysis for sample size during testing'
    
    def __init__(self, df) -> None:
        super().__init__(df)
        
    def plot_graph(self):
        for test_type in [1, 2]:
            df = self.df
        
            # Type
            df = df[df['te_tst_s1s2_type'].map(lambda x: x in (['hm', 'mh'] if test_type == 2 else ['hh', 'mm']))]
            
            # Add required rows
            for x in df.te_sample_size.unique():
                mean = df[df.te_sample_size == x]['test_power'].mean()
                row = {'te_sample_size': x, 'te_tst_datasets': 'Average', 'test_power': mean}
                row = pd.DataFrame(columns=df.columns, data=[row])
                df = pd.concat([df, row], axis=0)

            # Plot
            sns.set_style("whitegrid")
            g = sns.catplot(df, kind='point', x='te_sample_size', y='test_power', hue='te_tst_datasets', sharey=True, aspect=1.5, capsize=0.2, err_kws={'linewidth': 1})
                    
            # Add desired line
            GraphAnalysis.add_desired_line(g, 4.5, test_type)

            # Style
            g.despine(left=True)
            g.set_xlabels('Two-Sample Test Sample Set Size')
            g.set_ylabels(f'Type-{"I" if test_type == 1 else "II"} Rejection Rate (Mean)')
            g.legend.set_title('Test Dataset')
            plt.show()    
            

class GraphAnalysisSampleSizeAcrossModel(GraphAnalysis):
    description = 'Analysis for sample size during testing across models trained with different datasets'
    
    def __init__(self, df) -> None:
        super().__init__(df)
         
    def plot_graph(self):
        for test_dataset in ['TruthfulQA', 'SQuAD1', 'NarrativeQA', 'Average']:
            print(f"Printing graph for dataset {test_dataset}...")
            for test_type in [1, 2]:
                df = self.df
            
                # Filter Type
                df = df[df['te_tst_s1s2_type'].map(lambda x: x in (['hm', 'mh'] if test_type == 2 else ['hh', 'mm']))]
                
                # Add required rows
                for tr_datasets in df.tr_datasets.unique():
                    for x in df.te_sample_size.unique():
                        mean = df[(df.te_sample_size == x) & (df.tr_datasets == tr_datasets)]['test_power'].mean()
                        row = {'te_sample_size': x, 'tr_datasets': tr_datasets, 'te_tst_datasets': 'Average', 'test_power': mean}
                        row = pd.DataFrame(columns=df.columns, data=[row])
                        df = pd.concat([df, row], axis=0)
                
                # Filter Dataset
                df = df[df['te_tst_datasets'] == test_dataset]

                # Plot
                sns.set_style("whitegrid")
                g = sns.catplot(df, kind='point', x='te_sample_size', y='test_power', hue='tr_datasets', sharey=True, aspect=1.5, capsize=0.2, err_kws={'linewidth': 1})
                if test_type == 1:
                    g.set(ylim=(0.0, 0.3))

                # Add desired line
                GraphAnalysis.add_desired_line(g, 4.5, test_type)

                # Style
                g.despine(left=True)
                g.set_xlabels('Two-Sample Test Sample Set Size')
                g.set_ylabels(f'Type-{"I" if test_type == 1 else "II"} Rejection Rate (Mean)')
                g.legend.set_title('Model')
                plt.show()  


class GraphAnalysisSeed(GraphAnalysis):
    description = 'Analysis for models with different training seeds'
    
    def __init__(self, df) -> None:
        super().__init__(df)
        
    def plot_graph(self):
        for test_type in [1, 2]:
            df = self.df
        
            # Type
            df = df[df['te_tst_s1s2_type'].map(lambda x: x in (['hm', 'mh'] if test_type == 2 else ['hh', 'mm']))]
            
            # Add required rows
            for x in df.tr_seed.unique():
                mean = df[df.tr_seed == x]['test_power'].mean()
                row = {'tr_seed': x, 'te_tst_datasets': 'Average', 'test_power': mean}
                row = pd.DataFrame(columns=df.columns, data=[row])
                df = pd.concat([df, row], axis=0)

            # Plot
            sns.set_style("whitegrid")
            g = sns.catplot(df, kind='bar', x='tr_seed', y='test_power', hue='te_tst_datasets', sharey=True, aspect=1.5, capsize=0.2, err_kws={'linewidth': 1})

            # Add desired line
            GraphAnalysis.add_desired_line(g, 4.5, test_type)

            # Style
            g.despine(left=True)
            g.set_xlabels('Training Seed')
            g.set_ylabels(f'Type-{"I" if test_type == 1 else "II"} Rejection Rate (Mean)')
            g.legend.set_title('Test Dataset')
            plt.show()  
                

class GraphAnalysisSSTTrueRatio(GraphAnalysis):
    description = 'Analysis for single sample test true data ratio'
    
    def __init__(self, df) -> None:
        super().__init__(df)
        
    def plot_graph(self):
        for test_type in [1, 2]:
            df = self.df
        
            # Filter Type
            df = df[df['te_sst_userfill_type'].map(lambda x: x in (['hm', 'mh'] if test_type == 2 else ['hh', 'mm']))]
            
            # Add required rows
            for x in df.te_sst_true_ratio.unique():
                mean = df[df.te_sst_true_ratio == x]['test_power'].mean()
                row = {'te_sst_true_ratio': x, 'te_sst_user_dataset': 'Average', 'test_power': mean}
                row = pd.DataFrame(columns=df.columns, data=[row])
                df = pd.concat([df, row], axis=0)

            # Plot
            sns.set_style("whitegrid")
            g = sns.catplot(df, kind='point', x='te_sst_true_ratio', y='test_power', hue='te_sst_user_dataset', sharey=True, aspect=1.5, capsize=0.2, err_kws={'linewidth': 1})
            if test_type == 1:
                g.set(ylim=(0.0, 0.3))

            # Add desired line
            GraphAnalysis.add_desired_line(g, 4.5, test_type)

            # Style
            g.despine(left=True)
            g.set_xlabels('Single-Sample Test True User Data Ratio')
            g.set_ylabels(f'Type-{"I" if test_type == 1 else "II"} Rejection Rate (Mean)')
            g.legend.set_title('Test Dataset')
            plt.show()  


###########################################
# Analysis access points
###########################################

S1S2 = (('hh', 'mm'), ('hm', 'mh'))
ARG_TO_ANALYSIS = {
    'tabular_seed': TabularAnalysisSeed,
    'tabular_shuffle': TabularAnalysisShuffle,
    'tabular_linearSize': TabularAnalysisSize,
    'tabular_sampleSize': TabularAnalysisSampleSize,
    'tabular_permutationCount': TabularAnalysisPermutationCount,
    'tabular_LLM': TabularAnalysisLLM,
    'tabular_LLMSampleSize': TabularAnalysisLLMSampleSize,
    'tabular_SSTTrueDataRatio': TabularAnalysisSSTTrueDataRatio,
    'tabular_SSTDataset': TabularAnalysisSSTDataset,
    'tabular_SSTOptimalLLM': TabularAnalysisSSTOptimalLLM,
    'graphic_seed': GraphAnalysisSeed,
    'graphic_shuffle': GraphAnalysisShuffle,
    'graphic_linearSize': GraphAnalysisLinearSize,
    'graphic_sampleSize': GraphAnalysisSampleSize,
    'graphic_sampleSizeAcrossModel': GraphAnalysisSampleSizeAcrossModel,
    'graphic_permutationCount': GraphAnalysisPermCnt,
    'graphic_permutationCountTiming': GraphAnalysisPermCntTiming,
    'graphic_SSTTrueDataRatio': GraphAnalysisSSTTrueRatio,
}

def get_args():
    parser = ArgumentParser()
    
    parser.add_argument('--csv_files', type=str, nargs='+', help='CSV files to analyze, files are merged into one for analysis.')
    parser.add_argument('--analysis_name', type=str, choices=ARG_TO_ANALYSIS.keys(), help='Analysis to run.')
    
    parser.add_argument('--test_log_path', type=str, default='./test_logs/', required=False, help='Directory to test logs.')
    parser.add_argument('--output_folder', type=str, default='./analysis_logs/', required=False, help='Directory to output analysis logs.')
    parser.add_argument('--debug', action='store_true', required=False, help='Enable debug model to supress log file creation.')
    
    args = parser.parse_args()
    return vars(args) # return dict
  
def main():
    start_time_str = util.get_current_time_str()
    
    args = get_args()
    
    # Setup logs
    logger = util.setup_logs(
        file_path=None,
        id=start_time_str,
        is_debug=args['debug'],
        supress_file=True,
    )
    
    # Get data
    logger.info(f"Using files: {args['csv_files']}")
    df_list = []
    for f in args['csv_files']:
        df_list.append(pd.read_csv(os.path.join(args['test_log_path'], f)))
    df = pd.concat(df_list, axis=0)
    
    # Start analysis
    analysis = ARG_TO_ANALYSIS[args['analysis_name']](df)

    if isinstance(analysis, TabularAnalysis):
        logger.info(f'Running Tabular Analysis: {analysis.description}...')
        result = analysis.get_postprocessed_df(logger)
        
        # Save to csv
        if not args['debug']:
            output_csv_path = os.path.join(args['output_folder'], f'analysis_{start_time_str}.csv')
            result.to_csv(output_csv_path)
            logger.info(f'Result saved to csv file at {output_csv_path}')
            
    elif isinstance(analysis, GraphAnalysis):
        logger.info(f'Running Graph Analysis: {analysis.description}...')
        analysis.plot_graph()

if __name__ == '__main__':
    main()