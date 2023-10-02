import os
from argparse import ArgumentParser

from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt


def save_plots(model_dir, model_names):
    for mn in model_names:
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

        # Train J graph
        plt.figure()
        train_j_g = sns.lineplot(data=df, x='step', y='Train/J_stars')
        train_j_g.set(xlabel='Step', ylabel='J*', title='Train J*')
        # plt.ylim([-500, 10]) # Limit the y-axis
        plt.savefig(os.path.join(fig_dir, 'train_j.png'))

        # Train mmd
        plt.figure()
        train_mmd_g = sns.lineplot(data=df, x='step', y='Train/mmd_values')
        train_mmd_g.set(xlabel='Step', ylabel='MMD', title='Train MMD')
        plt.savefig(os.path.join(fig_dir, 'train_mmd.png'))
        
        # Train mmd std
        plt.figure()
        train_mmd_g = sns.lineplot(data=df, x='step', y='Train/mmd_stds')
        train_mmd_g.set(xlabel='Step', ylabel='MMD', title='Train MMD std. dev.')
        plt.savefig(os.path.join(fig_dir, 'train_mmd_std.png'))
        
        # Train power
        plt.figure()
        train_power_g = sns.lineplot(data=df, x='step', y='Train/train_power')
        train_power_g.set(xlabel='Step', ylabel='Power', title='Train Power (Type 2)')
        plt.savefig(os.path.join(fig_dir, 'train_power.png'))
        
        # Validation mmd
        plt.figure()
        val_mmd_g = sns.lineplot(data=df, x='step', y='Validation/val_mmd_diff_10')
        val_mmd_g.set(xlabel='Step', ylabel='MMD', title='Validation MMD')
        plt.savefig(os.path.join(fig_dir, 'val_mmd.png'))
        
        # Validation type 1 power
        plt.figure()
        val_power_1_g = sns.lineplot(data=df, x='step', y='Validation/val_power_same_10')
        val_power_1_g.set(xlabel='Step', ylabel='Power', title='Validation Power (Type 1)')
        plt.savefig(os.path.join(fig_dir, 'val_power_1.png'))
        
        # Validation type 2 power
        plt.figure()
        val_power_2_g = sns.lineplot(data=df, x='step', y='Validation/val_power_diff_10')
        val_power_2_g.set(xlabel='Step', ylabel='Power', title='Validation Power (Type 2)')
        plt.savefig(os.path.join(fig_dir, 'val_power_2.png'))

def get_args():
    # Setup parser
    parser = ArgumentParser()
    # Per Run Parameters
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--model_names', nargs='+', type=str)

    args = parser.parse_args()
    return vars(args) # return dict

def main():
    args = get_args()
    save_plots(**args)
    
if __name__ == '__main__':
    main()