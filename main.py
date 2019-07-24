import argparse
from solvers.BaselineSolver import BaselineSolver
from solvers.DANNSolver import DANNSolver
from solvers.MTSolver import MTSolver
import os

print(os.getcwd())
os.chdir(os.getcwd())

parser = argparse.ArgumentParser(description='Hello')

parser.add_argument('--model', type=str, default='DANN')
parser.add_argument('--dataset', type=str, default='Office31')
parser.add_argument('--source', type=str, default='Webcam')
parser.add_argument('--target', type=str, default='Dslr')

parser.add_argument('--test_mode', action='store_true', default=False)
parser.add_argument('--pretrained', action='store_true', default=False)
parser.add_argument('--if_test', action='store_true', default=False)
parser.add_argument('--cuda', type=str, default='cuda:0')

parser.add_argument('--batch_size', type=int, default=36)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--epochs', type=int, default=999999)
parser.add_argument('--iterations', type=int, default=999999)
parser.add_argument('--test_interval', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.001)

args = parser.parse_args()


def main():
    solver = None

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
            gamma=args.gamma
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
            gamma=args.gamma
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
            gamma=args.gamma
        )

    solver.solve()


if __name__ == '__main__':
    main()
