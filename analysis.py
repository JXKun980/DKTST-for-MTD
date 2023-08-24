import os
import itertools as it

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
    'Train - Batch Size': 'tr_bat_size',
    'Test - Dataset Name': 'te_ds_name',
    'Test - Dataset LLM Name': 'te_ds_llm_name',
    'Test - S1 S2 Type': 'te_s1s2',
    'Test - Significance Level': 'te_sig_lvl',
    'Test - Permutation Count': 'te_perm_cnt',
    'Test - Shuffled': 'te_shuffled',
    'Test - Batch Size': 'te_bat_size',
    'Test - Seed': 'te_seed',
    'Test - Test Size': 'test_size',
    'Result - Test Power': 'test_power',
    'Result - Threshold (Avg.)': 'threshold',
    'Result - MMD (Avg.)': 'mmd'
}
VAR_NAME_TO_CSV_NAME = {v: k for k, v in CSV_NAME_TO_VAR_NAME.items()}
DATA_PATH = './test_logs/'
S1S2 = (('hh', 'mm'), ('hm'))
    
def get_data(file_name):
    data_file = file_name
    df = pd.read_csv(os.path.join(DATA_PATH, data_file))
    return df

def get_preprocessed_df(df):
    df_new = df.drop(columns=df.columns[0])
    df_new = df_new.rename(columns=CSV_NAME_TO_VAR_NAME)
    return df_new

def get_analysis_groupby_database(df, analysis_fun, datasets=DATASETS):
    '''Get analysis result for each database and a result for the aggregated data, combine the results horizontally.'''
    if len(datasets) < 1: raise ValueError("Database list must at least contain one dataset")
    
    # Get result for each database, and then concatenate together in axis 1.
    result = None
    for ds in datasets:
        result_ds = analysis_fun(df[df.te_ds_name == ds])
        result = np.concatenate((result, result_ds), axis=1) if result is not None else result_ds
        
    # Get result for the aggregated database
    temp = analysis_fun(df)
    result = np.concatenate((result, temp), axis=1)
    return result

def get_general_shuffled_analysis(df, shuffled_tr=(True, False), shuffled_te=(True, False)):
    # Function for getting result from a single database (closure)
    def general_shuffled_analysis(df: pd.DataFrame):
        result = np.zeros((2, 4))
        for i, (trs, tes) in enumerate(it.product(shuffled_tr, shuffled_te)):
            for j, s1s2 in enumerate(S1S2):
                result[j, i] = df[
                        (df.tr_shuffled == trs) 
                        & (df.te_shuffled == tes) 
                        & (df.te_s1s2.map(lambda x: x in s1s2))
                    ].test_power.mean()
        return result
    
    # Get result for each database, and then concatenate together in axis 1.
    result = get_analysis_groupby_database(df=df, analysis_fun=general_shuffled_analysis, datasets=DATASETS)
    
    # Convert to dataframe table
    df_result = pd.DataFrame(result,
                index=pd.Index(('Type 1', 'Type 2'), name='Test Power:'),
                columns=pd.MultiIndex.from_product(
                    (
                        DATASETS+['Aggregated'], 
                        shuffled_tr, 
                        shuffled_te
                    ), 
                    names=('Database', 'Train Shuffled:', 'Test Shuffled:')
                ))
    return df_result



def get_general_lin_size_analysis(df, lin_sizes=None):
    lin_sizes= df.linear_size.unique() if lin_sizes is None else lin_sizes
    # Function for getting result from a single database
    def general_lin_size_analysis(df: pd.DataFrame):
        result = np.zeros((2, len(lin_sizes)))
        for i, ls in enumerate(lin_sizes):
            for j, s1s2 in enumerate(S1S2):
                result[j, i] = df[
                        (df.linear_size == ls) 
                        & (df.te_s1s2.map(lambda x: x in s1s2))
                    ].test_power.mean()
        return result

    # Get result for each database, and then concatenate together in axis 1.
    result = get_analysis_groupby_database(df=df, analysis_fun=general_lin_size_analysis, datasets=DATASETS)
    
    # Convert to dataframe table
    df_result = pd.DataFrame(result,
                index=pd.Index(('Type 1', 'Type 2'), name='Test Power:'),
                columns=pd.MultiIndex.from_product(
                    (
                        DATASETS+['Aggregated'], 
                        lin_sizes
                    ), 
                    names=('Database', 'Linear Size Multiple:')
                ))
    return df_result

def get_general_test_bat_size_analysis(df, batch_sizes=None):
    batch_sizes= df.te_bat_size.unique() if batch_sizes is None else batch_sizes
    # Function for getting result from a single database
    def general_test_bat_size_analysis(df: pd.DataFrame):
        result = np.zeros((2, len(batch_sizes)))
        for i, bs in enumerate(batch_sizes):
            for j, s1s2 in enumerate(S1S2):
                result[j, i] = df[
                        (df.te_bat_size == bs) 
                        & (df.te_s1s2.map(lambda x: x in s1s2))
                    ].test_power.mean()
        return result

    # Get result for each database, and then concatenate together in axis 1.
    result = get_analysis_groupby_database(df=df, analysis_fun=general_test_bat_size_analysis, datasets=DATASETS)
    
    # Convert to dataframe table
    df_result = pd.DataFrame(result,
                index=pd.Index(('Type 1', 'Type 2'), name='Test Power:'),
                columns=pd.MultiIndex.from_product(
                    (
                        DATASETS+['Aggregated'], 
                        batch_sizes
                    ), 
                    names=('Database', 'Test Batch Size:')
                ))
    return df_result


def get_general_perm_cnt_analysis(df, perm_cnts=None):
    perm_cnts = df.te_perm_cnt.unique() if perm_cnts is None else perm_cnts
    
    # Function for getting result from a single database
    def general_perm_cnt_analysis(df: pd.DataFrame):
        result = np.zeros((2, len(perm_cnts)))
        for i, pc in enumerate(perm_cnts):
            for j, s1s2 in enumerate(S1S2):
                result[j, i] = df[
                        (df.te_perm_cnt == pc) 
                        & (df.te_s1s2.map(lambda x: x in s1s2))
                    ].test_power.mean()
        return result

    # Get result for each database, and then concatenate together in axis 1.
    result = get_analysis_groupby_database(df=df, analysis_fun=general_perm_cnt_analysis, datasets=DATASETS)
    
    # Convert to dataframe table
    df_result = pd.DataFrame(result,
                index=pd.Index(('Type 1', 'Type 2'), name='Test Power:'),
                columns=pd.MultiIndex.from_product(
                    (
                        DATASETS+['Aggregated'], 
                        perm_cnts
                    ), 
                    names=('Database', 'Test Permutation Count:')
                ))
    return df_result

def get_general_other_LLM_analysis(df, perm_cnts=None):
    pass

def get_general_true_data_ratio_analysis():
    pass



def get_proportion_analysis_for_acc_above(df, acc_threshold=0.9):
    '''Return a dict containing (param_name : result_dataframe) pairs'''
    df_filtered = df[(df.test_power >= acc_threshold)]
    full_results = {}
    for col_name, col_series in df_filtered.items():
        value_ratio = col_series.value_counts(normalize=True)
        if value_ratio.count() > 1 and col_name not in ("model_name", "test_power", "threshold", "mmd"):
            result = pd.DataFrame(
                [value_ratio.tolist()], 
                index=pd.Index(('Percentage',)),
                columns=pd.MultiIndex.from_product(
                (
                    [col_name],
                    value_ratio.index.tolist()
                ),
                names=("Parameter", "Paramter Value")
            ))
            full_results[col_name] = result
    return full_results
            

def param_acc_analysis():
    start_time_str = util.get_current_time_str()
    test_file_name = 'test_20230817055948.csv'
    logger = util.setup_logs(
        file_path=f"./analysis_logs/analysis_{start_time_str}.log",
        id=start_time_str,
    )
    
    # Get data
    logger.info(f"Getting and pre-processing data from file {test_file_name}")
    df = get_data(test_file_name)
    df = get_preprocessed_df(df)
    
    # Analysis from existing data
    logger.info("Starting analysis...")
    gsa_df = get_general_shuffled_analysis(df)
    glsa_df = get_general_lin_size_analysis(df)
    gtbsa_df = get_general_test_bat_size_analysis(df)
    gpca_df = get_general_perm_cnt_analysis(df)
    
    # Log results
    logger.info(
        "\nResult of Shuffled Data Analysis:\n"
        f"{gsa_df}\n")
    
    logger.info(
        "\nResult of Linear Layer Size Analysis:\n"
        f"{glsa_df}\n")
    
    logger.info(
        "\nResult of Test Batch Size Analysis:\n"
        f"{gtbsa_df}\n")
    
    logger.info(
        "\nResult of Test Permutation Count Analysis:\n"
        f"{gpca_df}\n")
    
def param_acc_filtered_analysis():
    '''Test the accuracy by varying only a subset of paramters'''
    start_time_str = util.get_current_time_str()
    test_file_name = 'test_20230817055948.csv'
    
    logger = util.setup_logs(
        file_path=f"./analysis_logs/analysis_{start_time_str}.log",
        id=start_time_str,
    )
    
    # Get data
    logger.info(f"Getting and pre-processing data from file {test_file_name}")
    df = get_data(test_file_name)
    df = get_preprocessed_df(df)
    
    # Analysis from existing data
    logger.info("Starting analysis...")
    
    gtbsa_df = get_general_test_bat_size_analysis(df)
    logger.info(
        "\nResult of Test Batch Size Analysis:\n"
        f"{gtbsa_df}\n")
    
    gpca_df = get_general_perm_cnt_analysis(df)
    logger.info(
        "\nResult of Test Permutation Count Analysis:\n"
        f"{gpca_df}\n")

def database_best_acc_analysis():
    '''Using best paramter values and only test the batch size and permutation count parameters'''
    start_time_str = util.get_current_time_str()
    test_file_name = 'test_20230817055948.csv'
    
    logger = util.setup_logs(
        file_path=f"./analysis_logs/analysis_{start_time_str}.log",
        id=start_time_str,
    )
    
    # Get data
    logger.info(f"Getting and pre-processing data from file {test_file_name}")
    df = get_data(test_file_name)
    df = get_preprocessed_df(df)
    
    # Filter data to choose the best param
    df_filterd = df[
        (df.linear_size == 5)
        & (df.tr_shuffled == False)
        & (df.te_shuffled == True)
    ]
    
    # Analysis from existing data
    logger.info("Starting analysis...")

    gtbsa_df = get_general_test_bat_size_analysis(df_filterd)
    logger.info(
        "\nResult of Best Accuracy Analysis for different Test Batch Size:\n"
        f"{gtbsa_df}\n")
    
    gpca_df = get_general_perm_cnt_analysis(df_filterd)
    logger.info(
        "\nResult of Best Accuracy Analysis for different Test Permutation Count:\n"
        f"{gpca_df}\n")

def param_proportion_acc_analysis():
    '''Accuracy analysis that instead focus on proporation of a certain paramter value over a specific accuracy threshold'''
    start_time_str = util.get_current_time_str()
    test_file_name = 'test_20230817055948.csv'
    
    logger = util.setup_logs(
        file_path=f"./analysis_logs/analysis_{start_time_str}.log",
        id=start_time_str,
    )
    
    # Get data
    logger.info(f"Getting and pre-processing data from file {test_file_name}")
    df = get_data(test_file_name)
    df = get_preprocessed_df(df)
    
    # Analysis from existing data
    acc_threshold = 0.85
    logger.info(f"Starting analysis for type 2 accuracy above {acc_threshold}")
    results = get_proportion_analysis_for_acc_above(df, acc_threshold=acc_threshold)
    
    # Log data
    for r in results.values():
        logger.info(f"\n{r}\n")
    
if __name__ == '__main__':
    # param_acc_analysis()
    # param_acc_filtered_analysis()
    # database_best_acc_analysis()
    # param_proportion_acc_analysis()
    pass