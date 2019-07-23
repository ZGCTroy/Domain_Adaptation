import pandas as pd
import os

# 显示所有列
pd.set_option('display.max_columns', 10)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 400)
pd.set_option('expand_frame_repr', False)


def show_result(server, dataset, task, model):
    root_dir = '/Users/zhengguangcong/Desktop/logs/' + str(server)
    path = root_dir + '/' + dataset + '/' + task + '/' + model + '.csv'
    data = pd.read_csv(path, header=0)
    if dataset == 'Digits':
        sort_data = data.sort_values(['val_acc', 'test_acc'], inplace=False, ascending=False)
        sort_data = data.sort_values(['val_loss'], inplace=False, ascending=True)
       # sort_data = sort_data.sort_values(['test_acc'], inplace=False, ascending=False)
    else:
        sort_data = data.sort_values(['train_acc', 'test_acc'], inplace=False, ascending=False)

    print(sort_data.head(5))
    print()

def main():
    # Digits
    for task in ['MtoU']:
        for model in ['Baseline', 'DANN']:
            show_result(
                server=146,
                dataset='Digits',
                task=task,
                model=model
            )

if __name__ == '__main__':
    main()

