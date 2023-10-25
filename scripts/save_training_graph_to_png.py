import os
from argparse import ArgumentParser

from tbparse import SummaryReader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def save_plots(model_dir, model_names):
    if model_names is None:
        model_names = os.listdir(model_dir)
        for mn in model_names: # Remove invalid folders
            if 'train_config.yml' not in os.listdir(os.path.join(model_dir, mn)):
                model_names.remove(mn)
        
    for mn in tqdm(model_names):
        log_dir = os.path.join(model_dir, mn)
        reader = SummaryReader(log_dir, pivot=True)
        df = reader.scalars
        df = df.applymap(lambda x: x[-1] if type(x) == list else x) # Merge multiple log files with the last file's content
        df['Train/mmd_values'] = -df['Train/mmd_values']
        df = df.head(3000)
        
        # Create figures directory if it doesn't exist
        fig_dir = os.path.join(log_dir, 'figures')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        
        # (Optional) Fix legacy naming
        df = df.rename(columns={'Validation/val_mmd_diff_10': 'Validation/val_mmd_mean_diff_10'})
        
        # Smoothing
        df['Train/J_stars'] = gaussian_filter1d(df['Train/J_stars'], sigma=10)
        df['Train/mmd_values'] = gaussian_filter1d(df['Train/mmd_values'], sigma=10)
        df['Train/mmd_stds'] = gaussian_filter1d(df['Train/mmd_stds'], sigma=10)
        
        # Change all font sizes
        sns.set(font_scale=2)

        # Train J graph
        train_j_g = sns.lineplot(data=df, x='step', y='Train/J_stars')
        train_j_g.set(xlabel='Step', ylabel='J*', title='Train J*')
        # plt.ylim([-500, 10]) # Limit the y-axis
        plt.savefig(os.path.join(fig_dir, 'train_j.png'), bbox_inches='tight')
        plt.close()

        # Train mmd
        train_mmd_g = sns.lineplot(data=df, x='step', y='Train/mmd_values')
        train_mmd_g.set(xlabel='Step', ylabel='MMD', title='Train MMD')
        plt.savefig(os.path.join(fig_dir, 'train_mmd.png'), bbox_inches='tight')
        plt.close()
        
        # Train mmd std
        train_mmd_g = sns.lineplot(data=df, x='step', y='Train/mmd_stds')
        train_mmd_g.set(xlabel='Step', ylabel='MMD', title='Train MMD std. dev.')
        plt.savefig(os.path.join(fig_dir, 'train_mmd_std.png'), bbox_inches='tight')
        plt.close()
        
        # Train power
        train_power_g = sns.lineplot(data=df, x='step', y='Train/train_power')
        train_power_g.set(xlabel='Step', ylabel='Rejection Rate', title='Train Rejection Rate (Type 2)')
        plt.savefig(os.path.join(fig_dir, 'train_power.png'), bbox_inches='tight')
        plt.close()
        
        # Validation mmd
        val_mmd_g = sns.lineplot(data=df, x='step', y='Validation/val_mmd_mean_diff_10')
        val_mmd_g.set(xlabel='Step', ylabel='MMD', title='Validation MMD')
        plt.savefig(os.path.join(fig_dir, 'val_mmd.png'), bbox_inches='tight')
        plt.close()
        
        # Validation type 1 power
        val_power_1_g = sns.lineplot(data=df, x='step', y='Validation/val_power_same_10')
        val_power_1_g.set(xlabel='Step', ylabel='Rejection Rate', title='Validation Rejection Rate (Type 1)')
        plt.savefig(os.path.join(fig_dir, 'val_power_1.png'), bbox_inches='tight')
        plt.close()
        
        # Validation type 2 power
        val_power_2_g = sns.lineplot(data=df, x='step', y='Validation/val_power_diff_10')
        val_power_2_g.set(xlabel='Step', ylabel='Rejection Rate', title='Validation Rejection Rate (Type 2)')
        plt.savefig(os.path.join(fig_dir, 'val_power_2.png'), bbox_inches='tight')
        plt.close()

def get_args():
    # Setup parser
    parser = ArgumentParser()
    # Per Run Parameters
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory where models are saved.')
    parser.add_argument('--model_names', nargs='+', type=str, help='Name of the model folders to save plots for. If left out, all models in model_dir will be used.')

    args = parser.parse_args()
    return vars(args) # return dict

def main():
    args = get_args()
    save_plots(**args)
    
if __name__ == '__main__':
    main()