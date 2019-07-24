import pandas as pd
import os

# 显示所有列
pd.set_option('display.max_columns', 10)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 400)
pd.set_option('expand_frame_repr', False)


def show_result(dataset, task, model):
    root_dir = './true_logs/'
    path = root_dir + '/' + dataset + '/' + task + '/' + model + '.csv'
    data = pd.read_csv(path, header=0)
    if dataset == 'Digits':
        sort_data = data.sort_values(['val_acc', 'test_acc'], inplace=False, ascending=False)
        #sort_data = data.sort_values(['val_loss'], inplace=False, ascending=True)
    else:
        sort_data = data.sort_values(['train_acc','test_acc'], inplace=False, ascending=False)
        #sort_data = data.sort_values(['test_acc'], inplace=False, ascending=False)
        #sort_data = data.sort_values(['train_loss'], inplace=False, ascending=True)

    print(sort_data.head(3))
    print()


def main():
    # Digits
    print('Digits\n')
    for task in ['MtoU', 'UtoM', 'StoM']:
        for model in ['Baseline', 'DANN']:
            show_result(
                dataset='Digits',
                task=task,
                model=model
            )

    print('\n\n\nOffice31\n')

    # Office31
    for task in ['AtoW','AtoD','WtoA','WtoD','DtoW']:
        for model in ['Baseline','DANN']:
            show_result(
                dataset='Office31',
                task=task,
                model=model
            )

    print('\n\n\nOfficeHome\n')
    # OfficeHome
    for task in ['ArtoCl','ArtoPr','ArtoRe','CltoAr']:
        for model in ['Baseline','DANN']:
            show_result(
                dataset='OfficeHome',
                task=task,
                model=model
            )


if __name__ == '__main__':
    main()
