import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', 20)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 400)
pd.set_option('expand_frame_repr', False)


def show_result(dataset, task, model):
    root_dir = '../logs/'
    path = root_dir + dataset + '/' + task + '/' + model + '.csv'
    if os.path.exists(path):
        data = pd.read_csv(path, header=0)
        if dataset == 'Digits':
            sort_data = data.sort_values(['test_acc'], inplace=True, ascending=False)


def get_result():
    models = ['Baseline', 'MT', 'DANN', 'MCD', 'MADA']
    names = ['Baseline', 'MT+CT+TF', 'DANN', 'MCD', 'MADA']

    tasks = ['AtoW', 'DtoW', 'WtoD', 'AtoD', 'DtoA', 'WtoA', 'Avg']

    data = {
        task: [] for task in tasks
    }

    for model in models:
        y = []
        for task in tasks:
            path = '../logs/Office31/' + task + '/' + model + '.csv'
            if os.path.exists(path):
                tmp_data = pd.read_csv(path, header=0)
                data[task].append(tmp_data['test_acc'].max())
                y.append(tmp_data['test_acc'].max())

        data['Avg'].append(np.average(y))

    df = pd.DataFrame(
        data=data,
        columns=tasks,
        index=names
    )

    print(df)

    tasks = ['UtoM', 'MtoU', 'StoM', 'Avg']

    data = {
        task: [] for task in tasks
    }

    for model in models:
        y = []
        for task in tasks:
            path = '../logs/Digits/' + task + '/' + model + '.csv'
            if os.path.exists(path):
                tmp_data = pd.read_csv(path, header=0)
                data[task].append(tmp_data['test_acc'].max())
                y.append(tmp_data['test_acc'].max())

        data['Avg'].append(np.average(y))

    df = pd.DataFrame(
        data=data,
        columns=tasks,
        index=names
    )

    print(df)


def plot_Digits():
    plt.figure(dpi=1200)
    plt.style.use("seaborn-whitegrid")
    plt.ylabel('Accuracy on Target Domain')
    plt.xlabel('Task')
    markers = ['^', '.', 'p', '*', '+']
    models = ['Baseline', 'MT', 'DANN', 'MCD', 'MADA']
    names = ['Baseline', 'MT+CT+TF', 'DANN', 'MCD', 'MADA']
    i = -1
    for model in models:
        i += 1
        x = []
        y = []

        for task in ['MtoU', 'UtoM', 'StoM']:
            path = '../logs/Digits/' + task + '/' + model + '.csv'
            if os.path.exists(path):
                data = pd.read_csv(path, header=0)
                x.append(task)
                y.append(data['test_acc'].max())

        x.append('Average')
        y.append(np.average(y))

        plt.plot(x, y, marker=markers[i])

    plt.legend(names)
    plt.ylim((0.75, 1))
    plt.title('Domain Adaptation on Digits')
    plt.savefig('../pictures/Digits_plot.png')

    plt.show()


def plot_Office31():
    plt.figure(dpi=1200)
    plt.style.use("seaborn-whitegrid")
    plt.ylabel('Accuracy on Target Domain')
    plt.xlabel('Task')
    markers = ['^', '.', 'p', '*', '+']
    models = ['Baseline', 'MT', 'DANN', 'MCD', 'MADA']
    names = ['Baseline', 'MT+CT+TF', 'DANN', 'MCD', 'MADA']
    tasks = ['AtoW', 'DtoW', 'WtoD', 'AtoD', 'DtoA', 'WtoA']

    i = -1
    for model in models:
        i += 1
        x = []
        y = []

        for task in tasks:
            path = '../logs/Office31/' + task + '/' + model + '.csv'
            if os.path.exists(path):
                data = pd.read_csv(path, header=0)
                x.append(task)
                y.append(data['test_acc'].max())

        x.append('Average')
        y.append(np.average(y))
        plt.plot(x, y, marker=markers[i])

    plt.legend(names)
    plt.ylim((0.6, 1.0))
    plt.title('Domain Adaptation on Office31')

    plt.savefig('../pictures/Office31_plot.png')
    plt.show()


def bar_Office31():
    plt.figure(dpi=1200)
    plt.style.use("seaborn-whitegrid")
    plt.ylabel('Average classification accuracy on Target Domain')
    plt.xlabel('Task')
    plt.ylim((0.7, 1.0))
    plt.title('Average Classification Accuracy on Office31')

    models = ['Baseline', 'MT', 'DANN', 'MCD', 'MADA']
    names = ['Baseline', 'MT+CT+TF', 'DANN', 'MCD', 'MADA']
    x = [1, 2, 3, 4, 5]
    width = 0.15

    # Origin Office31
    avg = [0.761, 0, 0.822, 0, 0.852]
    plt.bar([i - width for i in x], avg, width=width * 2)

    # My Office31
    tasks = ['AtoW', 'DtoW', 'WtoD', 'AtoD', 'DtoA', 'WtoA']
    avg = []
    for model in models:
        y = []

        for task in tasks:
            path = '../logs/Office31/' + task + '/' + model + '.csv'
            if os.path.exists(path):
                data = pd.read_csv(path, header=0)
                y.append(data['test_acc'].max())

        avg.append(np.average(y))

    plt.bar([i + width for i in x], avg, width=width * 2)

    plt.bar([i for i in x], [0 for i in x], tick_label=names)

    plt.legend(['Origin Paper', 'Reproduction'])
    plt.savefig('../pictures/Office31_bar.png')
    plt.show()


def bar_Digits():
    plt.figure(dpi=1200)
    plt.style.use("seaborn-whitegrid")
    plt.ylabel('Average classification accuracy on Target Domain')
    plt.xlabel('Task')
    plt.ylim((0.7, 1.0))
    plt.title('Average Classification Accuracy on Digits')
    plt.legend(['Origin Paper', 'Reproduction'])

    models = ['Baseline', 'MT', 'DANN', 'MCD', 'MADA']
    names = ['Baseline', 'MT+CT+TF', 'DANN', 'MCD', 'MADA']

    x = [1, 2, 3, 4, 5]
    width = 0.15

    # Origin Digits
    avg = [0, 0.985, 0, 0.9483, 0]
    plt.bar([i - width for i in x], avg, width=width * 2)

    # My Digits
    tasks = ['MtoU', 'UtoM', 'StoM']
    avg = []
    for model in models:
        y = []

        for task in tasks:
            path = '../logs/Digits/' + task + '/' + model + '.csv'
            if os.path.exists(path):
                data = pd.read_csv(path, header=0)
                y.append(data['test_acc'].max())

        avg.append(np.average(y))

    plt.bar([i + width for i in x], avg, width=width * 2)

    plt.bar([i for i in x], [0 for i in x], tick_label=names)

    plt.legend(['Origin Paper', 'Reproduction'])
    plt.savefig('../pictures/Digits_bar.png')
    plt.show()


def main():
    bar_Digits()
    bar_Office31()
    plot_Digits()
    plot_Office31()

    get_result()


if __name__ == '__main__':
    main()
