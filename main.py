import argparse
from solvers.BaselineSolver import BaselineSolver

import os
print(os.getcwd())
os.chdir(os.getcwd())

parser = argparse.ArgumentParser(description='Hello')

parser.add_argument('--dataset',type=str,default='Office')
parser.add_argument('--source', type=str, default='Webcam')
parser.add_argument('--target', type=str, default='Dslr')

parser.add_argument('--test_mode', action='store_true', default=False)
parser.add_argument('--pretrained', action='store_true', default=False)
parser.add_argument('--if_test', action='store_true', default=False)
parser.add_argument('--cuda', type=str, default='cuda:0')


parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--epochs', type=int, default=300)


args = parser.parse_args()

def main():
    solver = BaselineSolver(
        dataset_type = args.dataset,
        source_domain = args.source,
        target_domain = args.target,
        cuda=args.cuda,
        pretrained = args.pretrained,
        test_mode = args.test_mode,
        if_test= args.if_test,
        batch_size = args.batch_size,
        num_epochs = args.epochs,
        num_workers = args.num_workers
    )

    solver.solve()

if __name__ == '__main__':
    main()

