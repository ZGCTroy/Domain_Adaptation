import argparse
import os
import numpy as np
import torch
import random

from solvers.BaselineSolver import BaselineSolver
from solvers.DANNSolver import DANNSolver
from solvers.MADASolver import MADASolver
from solvers.MCDSolver import MCDSolver
from solvers.MYMCDSolver import MYMCDSolver
from solvers.MTSolver import MTSolver
from solvers.MYMADASolver import MYMADASolver

print(os.getcwd())
os.chdir(os.getcwd())

parser = argparse.ArgumentParser(description='Hello')

parser.add_argument('--model', type=str, default='DANN')
parser.add_argument('--dataset', type=str, default='Office31')
parser.add_argument('--source', type=str, default='Webcam')
parser.add_argument('--target', type=str, default='Dslr')
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--cuda', type=str, default='cuda:0')
parser.add_argument('--data_root_dir', type=str, default='./data')

parser.add_argument('--test_mode', action='store_true', default=False)
parser.add_argument('--pretrained', action='store_true', default=False)
parser.add_argument('--if_test', action='store_true', default=False)
parser.add_argument('--use_CT', action='store_true', default=False)
parser.add_argument('--use_augment', action='store_true', default=False)

parser.add_argument('--batch_size', type=int, default=36)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--epochs', type=int, default=999999)
parser.add_argument('--iterations', type=int, default=999999)
parser.add_argument('--test_interval', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=10)
parser.add_argument('--num_k', type=int, default=4)
parser.add_argument('--loss_weight', type=float, default=1.0)

args = parser.parse_args()

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    setup_seed(1029)

    solver = None

    if args.model == 'MYMADA':
        solver = MYMADASolver(
            dataset_type=args.dataset,
            source_domain=args.source,
            target_domain=args.target,
            cuda=args.cuda,
            pretrained=args.pretrained,
            test_mode=args.test_mode,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            test_interval=args.test_interval,
            max_iter_num=args.iterations,
            num_workers=args.num_workers,
            lr=args.lr,
            gamma=args.gamma,
            optimizer_type=args.optimizer,
            loss_weight=args.loss_weight,
            data_root_dir = args.data_root_dir
        )

    if args.model == 'Baseline':
        solver = BaselineSolver(
            dataset_type=args.dataset,
            source_domain=args.source,
            target_domain=args.target,
            cuda=args.cuda,
            pretrained=args.pretrained,
            test_mode=args.test_mode,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            test_interval=args.test_interval,
            max_iter_num=args.iterations,
            num_workers=args.num_workers,
            lr=args.lr,
            gamma=args.gamma,
            optimizer_type=args.optimizer,
            data_root_dir=args.data_root_dir
        )

    if args.model == 'DANN':
        solver = DANNSolver(
            dataset_type=args.dataset,
            source_domain=args.source,
            target_domain=args.target,
            cuda=args.cuda,
            pretrained=args.pretrained,
            test_mode=args.test_mode,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            test_interval=args.test_interval,
            max_iter_num=args.iterations,
            num_workers=args.num_workers,
            lr = args.lr,
            gamma=args.gamma,
            optimizer_type=args.optimizer,
            use_augment=args.use_augment,
            data_root_dir=args.data_root_dir
        )

    if args.model == 'MT':
        solver = MTSolver(
            dataset_type=args.dataset,
            source_domain=args.source,
            target_domain=args.target,
            cuda=args.cuda,
            pretrained=args.pretrained,
            test_mode=args.test_mode,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            test_interval=args.test_interval,
            max_iter_num=args.iterations,
            num_workers=args.num_workers,
            lr=args.lr,
            gamma=args.gamma,
            optimizer_type=args.optimizer,
            use_CT=args.use_CT,
            data_root_dir=args.data_root_dir
        )

    if args.model == 'MCD':
        solver = MCDSolver(
            dataset_type=args.dataset,
            source_domain=args.source,
            target_domain=args.target,
            cuda=args.cuda,
            pretrained=args.pretrained,
            test_mode=args.test_mode,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            test_interval=args.test_interval,
            max_iter_num=args.iterations,
            num_workers=args.num_workers,
            lr=args.lr,
            gamma=args.gamma,
            optimizer_type=args.optimizer,
            num_k=args.num_k,
            data_root_dir=args.data_root_dir
        )

    if args.model == 'MYMCD':
        solver = MYMCDSolver(
            dataset_type=args.dataset,
            source_domain=args.source,
            target_domain=args.target,
            cuda=args.cuda,
            pretrained=args.pretrained,
            test_mode=args.test_mode,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            test_interval=args.test_interval,
            max_iter_num=args.iterations,
            num_workers=args.num_workers,
            lr=args.lr,
            gamma=args.gamma,
            optimizer_type=args.optimizer,
            num_k=args.num_k,
            data_root_dir=args.data_root_dir
        )

    if args.model == 'MCD2':
        solver = MCD2Solver(
            dataset_type=args.dataset,
            source_domain=args.source,
            target_domain=args.target,
            cuda=args.cuda,
            pretrained=args.pretrained,
            test_mode=args.test_mode,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            test_interval=args.test_interval,
            max_iter_num=args.iterations,
            num_workers=args.num_workers,
            lr=args.lr,
            gamma=args.gamma,
            optimizer_type=args.optimizer,
            num_k=args.num_k,
            data_root_dir=args.data_root_dir
        )

    if args.model == 'MADA':
        solver = MADASolver(
            dataset_type=args.dataset,
            source_domain=args.source,
            target_domain=args.target,
            cuda=args.cuda,
            pretrained=args.pretrained,
            test_mode=args.test_mode,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            test_interval=args.test_interval,
            max_iter_num=args.iterations,
            num_workers=args.num_workers,
            lr=args.lr,
            gamma=args.gamma,
            optimizer_type=args.optimizer,
            loss_weight=args.loss_weight,
            data_root_dir=args.data_root_dir
        )

    solver.solve()


if __name__ == '__main__':
    main()
