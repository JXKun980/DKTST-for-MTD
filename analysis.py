import pandas as pd
import itertools as it
import numpy as np

csv_name_to_var_name = {
    'Train - Checkpoint Name': 'chkpnt_name',
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
    'Result - Test Power': 'test_power',
    'Result - Threshold (Avg.)': 'threshold',
    'Result - MMD (Avg.)': 'mmd'
}
var_name_to_csv_name = {v: k for k, v in csv_name_to_var_name.items()}
data_path = './test_logs/'

def main():
    # Get data
    df_dict = get_data()
    df_dict = get_preprocessed_df(df_dict)
    
    # Analysis from existing data
    gsa_df = get_general_shuffled_analysis(df_dict)
    glsa_df = get_general_lin_size_analysis(df_dict)
    gtbsa_df = get_general_test_bat_size_analysis(df_dict)
    
    # Analysis from new data
    df_dict_pc = {}
    df_dict_pc['SQuAD1'] = pd.read_csv(data_path + 'test_20230810065732 bat size perm cnt.csv')
    df_dict_pc = get_preprocessed_df(df_dict_pc)
    gpca_df = get_general_perm_cnt_analysis(df_dict_pc)
    print(gpca_df)
    
def get_data():
    data_file_dict = {
        'TruthfulQA': 'test_20230621060917 truthful QA.csv',
        'SQuAD1': 'test_20230724221201 SQuAD1.csv',
        'NarrativeQA': 'test_20230724221928 NarrativeQA.csv'
    }
    df_dict = {}
    for k, v in data_file_dict.items():
        df_dict[k] = pd.read_csv(data_path + v)
    return df_dict

def get_preprocessed_df(df_dict):
    df_dict_new = {}
    for db, df in df_dict.items():
        df_new = df.drop(columns=df.columns[0])
        df_new = df_new.rename(columns=csv_name_to_var_name)
        df_dict_new[db] = df_new
    return df_dict_new

def batch_analysis_util(df_dict: dict, analysis_fun):
    # Get result for each database, and then concatenate together in axis 1.
    result = None
    for df in df_dict.values():
        temp = analysis_fun(df)
        result = np.concatenate((result, temp), axis=1) if result is not None else temp
        
    # Get result for the aggregated database
    temp = analysis_fun(pd.concat((df for df in df_dict.values())))
    result = np.concatenate((result, temp), axis=1)
    return result

def get_general_shuffled_analysis(df_dict: dict):
    
    # Function for getting result from a single database
    def general_shuffled_analysis(df: pd.DataFrame):
        result = np.zeros((2, 4))
        for i, (trs, tes) in enumerate(it.product((True, False), (True, False))):
            for j, s1s2 in enumerate((('hh', 'mm'), ('hm'))):
                result[j, i] = df[
                        (df.tr_shuffled == trs) 
                        & (df.te_shuffled == tes) 
                        & (df.te_s1s2.map(lambda x: x in s1s2))
                    ].test_power.mean()
        return result
    
    # Get result for each database, and then concatenate together in axis 1.
    result = batch_analysis_util(df_dict, general_shuffled_analysis)
    
    # Convert to dataframe table
    df_result = pd.DataFrame(result,
                index=pd.Index(('Type 1', 'Type 2'), name='Test Power:'),
                columns=pd.MultiIndex.from_product(
                    (
                        tuple(df_dict.keys())+('Aggregated',), 
                        (True, False), 
                        (True, False)
                    ), 
                    names=('Database', 'Train Shuffled:', 'Test Shuffled:')
                ))
    return df_result



def get_general_lin_size_analysis(df_dict: dict):
    lin_sizes = (3, 5)
    
    # Function for getting result from a single database
    def general_lin_size_analysis(df: pd.DataFrame):
        result = np.zeros((2, len(lin_sizes)))
        for i, ls in enumerate(lin_sizes):
            for j, s1s2 in enumerate((('hh', 'mm'), ('hm'))):
                result[j, i] = df[
                        (df.linear_size == ls) 
                        & (df.te_s1s2.map(lambda x: x in s1s2))
                    ].test_power.mean()
        return result

    # Get result for each database, and then concatenate together in axis 1.
    result = batch_analysis_util(df_dict, general_lin_size_analysis)
    
    # Convert to dataframe table
    df_result = pd.DataFrame(result,
                index=pd.Index(('Type 1', 'Type 2'), name='Test Power:'),
                columns=pd.MultiIndex.from_product(
                    (
                        tuple(df_dict.keys())+('Aggregated',), 
                        lin_sizes
                    ), 
                    names=('Database', 'Linear Size Multiple:')
                ))
    return df_result

def get_general_test_bat_size_analysis(df_dict: dict):
    bat_sizes = (20, 10, 5, 4, 3)
    
    # Function for getting result from a single database
    def general_test_bat_size_analysis(df: pd.DataFrame):
        result = np.zeros((2, len(bat_sizes)))
        for i, bs in enumerate(bat_sizes):
            for j, s1s2 in enumerate((('hh', 'mm'), ('hm'))):
                result[j, i] = df[
                        (df.te_bat_size == bs) 
                        & (df.te_s1s2.map(lambda x: x in s1s2))
                    ].test_power.mean()
        return result

    # Get result for each database, and then concatenate together in axis 1.
    result = batch_analysis_util(df_dict, general_test_bat_size_analysis)
    
    # Convert to dataframe table
    df_result = pd.DataFrame(result,
                index=pd.Index(('Type 1', 'Type 2'), name='Test Power:'),
                columns=pd.MultiIndex.from_product(
                    (
                        tuple(df_dict.keys())+('Aggregated',), 
                        bat_sizes
                    ), 
                    names=('Database', 'Test Batch Size:')
                ))
    return df_result


def get_general_perm_cnt_analysis(df_dict: dict):
    perm_cnts = df_dict['SQuAD1'].te_perm_cnt.unique()
    
    # Function for getting result from a single database
    def general_perm_cnt_analysis(df: pd.DataFrame):
        result = np.zeros((2, len(perm_cnts)))
        for i, pc in enumerate(perm_cnts):
            for j, s1s2 in enumerate((('hh', 'mm'), ('hm'))):
                result[j, i] = df[
                        (df.te_perm_cnt == pc) 
                        & (df.te_s1s2.map(lambda x: x in s1s2))
                    ].test_power.mean()
        return result

    # Get result for each database, and then concatenate together in axis 1.
    result = batch_analysis_util(df_dict, general_perm_cnt_analysis)
    
    # Convert to dataframe table
    df_result = pd.DataFrame(result,
                index=pd.Index(('Type 1', 'Type 2'), name='Test Power:'),
                columns=pd.MultiIndex.from_product(
                    (
                        tuple(df_dict.keys())+('Aggregated',), 
                        perm_cnts
                    ), 
                    names=('Database', 'Test Permutation Count:')
                ))
    return df_result

if __name__ == '__main__':
    main()