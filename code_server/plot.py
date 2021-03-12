import argparse
from data_preprocessing import Data
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import json

def interval_baseline(stock):
    closing_path = os.path.join('./data_closing/' + stock + '-closing.json')
    states_path = os.path.join('./data_states/' + stock + '-states.json')
    data = Data(closing_path, states_path, 96)
    data.load()
    closing = data.closing
    period = 10
    current_portfolio = 100000
    sell_timestamp_diff = 5000 // period
    sell_volume_each_time = 10000

    baseline_plot = []
    baseline_plot.append(current_portfolio)

    for timestamp in range(period):
        last_timestamp_price = closing[timestamp * sell_timestamp_diff]
        current_timestamp_price = closing[(timestamp + 1) * sell_timestamp_diff]
        stock_price_diff = current_timestamp_price[1] - last_timestamp_price[1]
        current_portfolio += (stock_price_diff * sell_volume_each_time - 100)
        # print(current_portfolio)
        baseline_plot.append(current_portfolio)
    

    return baseline_plot

def plot_result(stocks):
    fig = plt.figure()
    for j in range(len(stocks)):
        ax = fig.add_subplot(3,4,j + 1)

        f1 = open('./results_server/results/portfolio/' + stocks[j] + '_portfolio.json')
        trained_result = json.load(f1)
        ax.plot(trained_result, label="result")

        f2 = open('./results_server/results_e/portfolio/' + stocks[j] + '_portfolio.json')
        baseline_result = json.load(f2)
        ax.plot(baseline_result, label="baseline-no-aug")

        f3 = open('./results_server/results_buy/portfolio/' + stocks[j] + '_portfolio.json')
        baseline_result2 = json.load(f3)
        f4 = open('./results_server/results_sell/portfolio/' + stocks[j] + '_portfolio.json')
        baseline_result3 = json.load(f4)
        if baseline_result2[-1] > baseline_result3[-1]: 
            ax.plot(baseline_result2, label="baseline-cont-buy(or sell, use better one)")
        else:
            ax.plot(baseline_result3, label="baseline-cont-buy(or sell, use better one)")

        baseline1 = interval_baseline(stocks[j])
        x_axis_interval = []
        for i in range(0, 5001, 5001 // (len(baseline1) - 1)):
            x_axis_interval.append(i)
        ax.plot(x_axis_interval, baseline1, label="baseline-interval")

        ax.plot(x_axis_interval, np.full(11, 100000), linestyle='dashed', label="100K$ Init. Val.")
        ax.axes.get_xaxis().set_ticks([0, 5000])
        ax.ticklabel_format(axis='y', style='sci',scilimits=(3,3))

        ax.set_title(stocks[j])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    fig.text(0.5, 0.04, 'Steps', ha='center')
    fig.text(0.04, 0.5, 'Portfolio (in thousand dollars)', va='center', rotation='vertical')
    plt.show()

def plot_ablation_gamma(stocks, gamma):
    fig = plt.figure()

    for j in range(len(stocks)):
        ax = fig.add_subplot(1,3,j + 1)

        # Gamma
        for m in range(len(gamma)):
            f1 = open('./results_ab/portfolio/' + stocks[j] + '_gamma_' + gamma[m] + '_portfolio.json')
            gamma_res = json.load(f1)
            ax.plot(gamma_res, label="Gamma: " + gamma[m])
        ax.plot(np.full(5000, 100000), linestyle='dashed', label="100K$ Init. Val.")

        ax.axes.get_xaxis().set_ticks([0, 5000])
        ax.ticklabel_format(axis='y', style='sci',scilimits=(3,3))

        ax.set_title(stocks[j])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    fig.text(0.5, 0.04, 'Steps', ha='center')
    fig.text(0.04, 0.5, 'Portfolio (in thousand dollars)', va='center', rotation='vertical')
    plt.show()

def plot_ablation_size(stocks, size_list):
    fig = plt.figure()

    for j in range(len(stocks)):
        ax = fig.add_subplot(1,3,j + 1)

        # Size
        for m in range(len(size_list)):
            f1 = open('./results_ab/portfolio/' + stocks[j] + size_list[m] + '.0000_portfolio.json')
            size_res = json.load(f1)
            ax.plot(size_res, label="Trade Size: " + size_list[m])
        ax.plot(np.full(5000, 100000), linestyle='dashed', label="100K$ Init. Val.")

        ax.axes.get_xaxis().set_ticks([0, 5000])
        ax.ticklabel_format(axis='y', style='sci',scilimits=(3,3))

        ax.set_title(stocks[j])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    fig.text(0.5, 0.04, 'Steps', ha='center')
    fig.text(0.04, 0.5, 'Portfolio (in thousand dollars)', va='center', rotation='vertical')
    plt.show()

def plot_ablation_spread(stocks, spread):
    fig = plt.figure()

    for j in range(len(stocks)):
        ax = fig.add_subplot(1,3,j + 1)

        # Spread
        for m in range(len(spread)):
            f1 = open('./results_ab/portfolio/' + stocks[j] + spread[m] + '_portfolio.json')
            spread_res = json.load(f1)
            ax.plot(spread_res, label="spread: " + spread[m])
        ax.plot(np.full(5000, 100000), linestyle='dashed', label="100K$ Init. Val.")

        ax.axes.get_xaxis().set_ticks([0, 5000])
        ax.ticklabel_format(axis='y', style='sci',scilimits=(3,3))

        ax.set_title(stocks[j])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    fig.text(0.5, 0.04, 'Steps', ha='center')
    fig.text(0.04, 0.5, 'Portfolio (in thousand dollars)', va='center', rotation='vertical')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline_trading')
    parser.add_argument('--stock', type=str, default='QCO', help="determine which stock")


    stock_list_1 = ['AAL', 'ABB', 'ABT', 'AIG','AMG', 'LMT', 'BAC', 'MSF','CAT', 'BII', 'COP', 'MPC']
    stock_list_2 = ['IBM', 'LUV', 'MA_', 'TRV','TSL', 'TMO', 'T_2', 'TWT','UNH', 'UAL', 'V_2', 'WMT']

    plot_result(stock_list_1)
    plot_result(stock_list_2)

    stock_list_3 = ['ABB', 'AMZ', 'MA_']
    gamma_list = ['0.0001', '0.0005', '0.0010', '0.0050', '0.0100', '0.1000']
    size_list = ['5000', '10000', '20000', '50000']
    spread_list = ['0.0001', '0.0005', '0.0010', '0.0050', '0.0100', '0.0500']

    plot_ablation_gamma(stock_list_3, gamma_list)
    plot_ablation_size(stock_list_3, size_list)
    plot_ablation_spread(stock_list_3, spread_list)


