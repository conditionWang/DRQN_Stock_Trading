import argparse
from data_preprocessing import Data
from trading_env import TradingEnv, RunAgent
from agent import Agent
from model import DQN
import torch
import numpy as np
import random
import os

def training(args):
    T = args.T
    M = args.bs # minibatch size
    alpha = args.lr # Learning rate
    gamma = args.gamma # Discount factor
    theta = args.para_target # Target network
    n_units = args.n_units # number of units in a hidden layer
    closing_path = os.path.join(args.root_path, 'data_closing/' + args.stock + '-closing.json')
    states_path = os.path.join(args.root_path, 'data_states/' + args.stock + '-states.json')

    RunAgent(TradingEnv(Data(closing_path, states_path, T)), Agent()).run(5000, args)

    # weight initialization!!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN_Trading')
    parser.add_argument('--root_path', type=str, default='./', help="root path")
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--bs', type=int, default=16, help="training batch size")
    parser.add_argument('--lr', type=float, default=0.00025, help="training learning rate")
    parser.add_argument('--gamma', type=float, default=0.001, help="the discount factor of Q learning")
    parser.add_argument('--n_units', type=int, default=32, help="the number of units in a hidden layer")
    parser.add_argument('--T', type=int, default=96, help="the length of series data")
    parser.add_argument('--stock', type=str, default='AIG', help="determine which stock")
    parser.add_argument('--seed', type=int, default=2038, help="random seed")
    parser.add_argument('--para_target', type=float, default=0.001, help="the parameter which controls the soft update")

    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    file_path = './data_closing/'
    datafile_list = os.listdir(file_path)
    for i in range(0, len(datafile_list)):
        args.stock = datafile_list[i][0:3]
        if args.stock == 'AXP':
            continue
        print(args.stock)
        SEED = args.seed
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        training(args)
