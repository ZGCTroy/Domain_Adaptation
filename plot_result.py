import pandas as pd
import os
from matplotlib import pyplot as plt

# 显示所有列
pd.set_option('display.max_columns', 20)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 400)
pd.set_option('expand_frame_repr', False)


def show_result(dataset, task, model):
    root_dir = './true_logs/'
    path = root_dir + dataset + '/' + task + '/' + model + '.csv'
    if os.path.exists(path):
        data = pd.read_csv(path, header=0)
        if dataset == 'Digits':
            # sort_data = data.sort_values(['val_acc', 'test_acc'], inplace=False, ascending=False)
            # sort_data = data.sort_values(['val_loss'], inplace=False, ascending=True)
            sort_data = data.sort_values(['test_acc'], inplace=False, ascending=False)

        else:
            # sort_data = data.sort_values(['train_acc','test_acc'], inplace=False, ascending=False)
            sort_data = data.sort_values(['test_acc'], inplace=False, ascending=False)
            # sort_data = data.sort_values(['train_loss'], inplace=False, ascending=True)

        print(sort_data.head(3))
        print()


def show_result_on_server(server, dataset, task, model):
    root_dir = '/Users/zhengguangcong/Desktop/logs/' + str(server)
    path = root_dir + '/' + dataset + '/' + task + '/' + model + '.csv'
    if os.path.exists(path):
        data = pd.read_csv(path, header=0)
        if dataset == 'Digits':
            sort_data = data.sort_values(['val_acc', 'test_acc'], inplace=False, ascending=False)
            sort_data = data.sort_values(['val_loss'], inplace=False, ascending=True)
            # sort_data = data.sort_values(['test_acc'], inplace=False, ascending=False)
        else:
            # sort_data = data.sort_values(['train_acc','test_acc'], inplace=False, ascending=False)
            # sort_data = data.sort_values(['test_acc'], inplace=False, ascending=False)
            sort_data = data.sort_values(['train_loss'], inplace=False, ascending=True)

        print(sort_data.head(3))
        print()


def plot_Digits():
    plt.figure(dpi=900)
    plt.rc('font', family='Times New Roman', size=13)
    plt.grid(linestyle='--')
    plt.ylabel('Accuracy on Target Domain')
    plt.xlabel('Task')
    markers = ['^', '.', 'p','*','+']
    models = ['Baseline', 'DANN','MT','MCD','MADA']
    i = -1
    for model in models:
        i += 1
        x = []
        y = []

        for task in ['MtoU', 'UtoM', 'StoM']:
            path = './true_logs/Digits/' + task + '/' + model + '.csv'
            if os.path.exists(path):
                data = pd.read_csv(path, header=0)
                x.append(task)
                y.append(data['test_acc'].max())

        print(y)
        plt.plot(x, y, marker=markers[i])

    plt.legend(models)
    plt.ylim((0.75, 1))

    plt.show()


def plot_Office31():
    plt.figure(dpi=900)
    plt.rc('font', family='Times New Roman', size=13)
    plt.grid(linestyle='--')
    plt.ylabel('Accuracy on Target Domain')
    plt.xlabel('Task')
    markers = ['^', '.', 'p','*','+']
    models = ['Baseline', 'DANN', 'MCD','MADA','MT']
    i = -1
    for model in models:
        i += 1
        x = []
        y = []

        for task in ['DtoA', 'WtoA', 'AtoW', 'AtoD', 'DtoW', 'WtoD']:
            path = './true_logs/Office31/' + task + '/' + model + '.csv'
            if os.path.exists(path):
                data = pd.read_csv(path, header=0)
                x.append(task)
                y.append(data['test_acc'].max())

        print(y)
        plt.plot(x, y, marker=markers[i])

    plt.legend(models)
    plt.ylim((0.6, 1.0))

    plt.show()


def plot_OfficeHome():
    plt.figure(dpi=900)
    plt.rc('font', family='Times New Roman', size=13)
    plt.grid(linestyle='--')
    plt.ylabel('Accuracy on Target Domain')
    plt.xlabel('Task')
    markers = ['^', '.', 'p','*','+']
    models = ['Baseline', 'DANN', 'MCD','MADA','MT']
    i = -1
    for model in models:
        i += 1
        x = []
        y = []

        for task in ['ArtoCl','CltoPr','RetoPw']:
            path = './true_logs/OfficeHome/' + task + '/' + model + '.csv'
            print(path)
            if os.path.exists(path):
                data = pd.read_csv(path, header=0)
                x.append(task)
                y.append(data['test_acc'].max())

        print(y)
        plt.plot(x, y, marker=markers[i])

    plt.legend(models)
    plt.ylim((0.3, 1.0))

    plt.show()

def main():
    plot_Digits()
    plot_Office31()
    plot_OfficeHome()

    # Digits
    # print('Digits\n')
    # for task in ['MtoU', 'UtoM', 'StoM']:
    #     for model in ['Baseline', 'DANN', 'MT','MCD']:
    #         show_result(
    #             dataset='Digits',
    #             task=task,
    #             model=model
    #         )

    #
    # print('\n\n\nOffice31\n')
    # #
    # Office31
    # for task in ['AtoD','AtoW','DtoA','DtoW','WtoA','WtoD']:
    #     for model in ['Baseline','DANN','MT']:
    #         show_result(
    #             dataset='Office31',
    #             task=task,
    #             model=model
    #         )
    #

    # print('\n\n\nOfficeHome\n')
    # # OfficeHome
    # for task in ['ArtoCl','ArtoPr','ArtoRe','CltoAr','CltoPr','CltoRe','PrtoAr','PrtoCl','PrtoRe','RetoAr','RetoCl','RetoPr']:
    #     for model in ['Baseline','DANN','MT']:
    #         show_result(
    #             dataset='OfficeHome',
    #             task=task,
    #             model=model
    #         )


if __name__ == '__main__':
    main()
