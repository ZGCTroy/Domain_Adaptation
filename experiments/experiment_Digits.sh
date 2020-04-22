#!/usr/bin/env bash

# Baseline

python3.6 ../main.py --model='Baseline' --dataset='Digits' --source='USPS' --target='MNIST' --cuda='cuda:0' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3 main.py --model='Baseline' --dataset='Digits' --source='MNIST' --target='USPS' \
--cuda='cuda:0' --num_workers=0 --epochs=99 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3.6 ../main.py --model='Baseline' --dataset='Digits' --source='SVHN' --target='MNIST' \
--cuda='cuda:0' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

# MY

python3 main.py --model='MY' --dataset='Digits' --source='MNIST' --target='USPS' \
--cuda='cuda:0' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3 main.py --model='MY' --dataset='Digits' --source='USPS' --target='MNIST' \
--cuda='cuda:0' --num_workers=3 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3 main.py --model='MY' --dataset='Digits' --source='SVHN' --target='MNIST' \
--cuda='cuda:0' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'


# DANN

python3 main.py --model='DANN' --dataset='Digits' --source='MNIST' --target='USPS' \
--cuda='cuda:0' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3.6 ../main.py --model='DANN' --dataset='Digits' --source='USPS' --target='MNIST' \
--cuda='cuda:0' --num_workers=3 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3 main.py --model='DANN' --dataset='Digits' --source='SVHN' --target='MNIST' \
--cuda='cuda:0' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'


# MT

python3.6 ../main.py --model='MT' --dataset='Digits' --source='USPS' --target='MNIST' \
--cuda='cuda:0' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3.6 ../main.py --model='MT' --dataset='Digits' --source='MNIST' --target='USPS' \
--cuda='cuda:1' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3.6 ../main.py --model='MT' --dataset='Digits' --source='SVHN' --target='MNIST' \
--cuda='cuda:2' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'



# MCD

python3.6 ../main.py --model='MCD' --dataset='Digits' --source='USPS' --target='MNIST' \
--cuda='cuda:0' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3 main.py --model='MCD' --dataset='Digits' --source='MNIST' --target='USPS' \
--cuda='cuda:0' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3.6 ../main.py --model='MCD' --dataset='Digits' --source='SVHN' --target='MNIST' \
--cuda='cuda:2' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

# MYMCD

python3.6 ../main.py --model='MCD' --dataset='Digits' --source='USPS' --target='MNIST' \
--cuda='cuda:0' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3 main.py --model='MYMCD' --dataset='Digits' --source='MNIST' --target='USPS' \
--cuda='cuda:0' --num_workers=0 --epochs=100 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3.6 ../main.py --model='MCD' --dataset='Digits' --source='SVHN' --target='MNIST' \
--cuda='cuda:2' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'



# MCD2

python3.6 ../main.py --model='MCD2' --dataset='Digits' --source='USPS' --target='MNIST' \
--cuda='cuda:0' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3.6 ../main.py --model='MCD2' --dataset='Digits' --source='MNIST' --target='USPS' \
--cuda='cuda:0' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

python3.6 ../main.py --model='MCD2' --dataset='Digits' --source='SVHN' --target='MNIST' \
--cuda='cuda:2' --num_workers=0 --epochs=300 --batch_size=256 --test_interval=1 --optimizer='Adam'

# MADA

python3 main.py --model='MADA' --dataset='Digits' --source='USPS' --target='MNIST' \
--cuda='cuda:2' --num_workers=0 --epochs=200 --batch_size=256 --test_interval=1 --optimizer='Adam' --loss_weight=0.5

python3 main.py --model='MY' --dataset='Digits' --source='USPS' --target='MNIST' \
--cuda='cuda:0' --num_workers=0 --epochs=200 --batch_size=256 --test_interval=1 --optimizer='Adam' --loss_weight=1.0

python3 main.py --model='MYMADA' --dataset='Digits' --source='MNIST' --target='USPS' \
--cuda='cuda:0' --num_workers=0 --epochs=99 --batch_size=256 --test_interval=1 --optimizer='Adam' --loss_weight=10.0


python3.6 ../main.py --model='MADA' --dataset='Digits' --source='SVHN' --target='MNIST' \
--cuda='cuda:0' --num_workers=0 --epochs=200 --batch_size=256 --test_interval=1 --optimizer='Adam' --loss_weight=0.5








