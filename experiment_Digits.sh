#!/usr/bin/env bash
# Baseline

python3.6 main.py --model='Baseline' --dataset='Digits' --source='USPS' --target='MNIST' \
--cuda='cuda:0' --num_workers=2 --epochs=300 --batch_size=256 --test_interval=1

python3.6 main.py --model='Baseline' --dataset='Digits' --source='MNIST' --target='USPS' \
--cuda='cuda:0' --num_workers=2 --epochs=300 --batch_size=256 --test_interval=1

python3.6 main.py --model='Baseline' --dataset='Digits' --source='SVHN' --target='MNIST' \
--cuda='cuda:0' --num_workers=2 --epochs=300 --batch_size=256 --test_interval=1


# DANN

python3.6 main.py --model='DANN' --dataset='Digits' --source='USPS' --target='MNIST' \
--cuda='cuda:0' --num_workers=2 --epochs=300 --batch_size=256 --test_interval=1

python3.6 main.py --model='DANN' --dataset='Digits' --source='MNIST' --target='USPS' \
--cuda='cuda:0' --num_workers=2 --epochs=300 --batch_size=256 --test_interval=1

python3.6 main.py --model='DANN' --dataset='Digits' --source='SVHN' --target='MNIST' \
--cuda='cuda:0' --num_workers=2 --epochs=300 --batch_size=256 --test_interval=1


# MT

python3.6 main.py --model='MT' --dataset='Digits' --source='USPS' --target='MNIST' \
--cuda='cuda:0' --num_workers=2 --epochs=300 --batch_size=256 --test_interval=1

python3.6 main.py --model='MT' --dataset='Digits' --source='MNIST' --target='USPS' \
--cuda='cuda:0' --num_workers=2 --epochs=300 --batch_size=256 --test_interval=1

python3.6 main.py --model='MT' --dataset='Digits' --source='SVHN' --target='MNIST' \
--cuda='cuda:0' --num_workers=2 --epochs=300 --batch_size=256 --test_interval=1











