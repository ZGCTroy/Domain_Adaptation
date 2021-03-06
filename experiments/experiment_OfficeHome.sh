# Baseline

python3.6 ../main.py --model='Baseline' --dataset='OfficeHome' --source='Art' --target='Product'\
 --cuda='cuda:1' --num_workers=1 --iterations=10004  --test_interval=100 --batch_size=36

python3.6 ../main.py --model='Baseline' --dataset='OfficeHome' --source='Art' --target='Real World' \
 --cuda='cuda:1' --num_workers=1 --iterations=10004  --test_interval=100 --batch_size=36

python3.6 ../main.py --model='Baseline' --dataset='OfficeHome' --source='Clipart' --target='Art' \
--cuda='cuda:1' --num_workers=1 --iterations=10004 --test_interval=100 --batch_size=36

python3.6 ../main.py --model='Baseline' --dataset='OfficeHome' --source='Clipart' --target='Product'\
 --cuda='cuda:1' --num_workers=1 --iterations=10004  --test_interval=100 --batch_size=36

python3.6 ../main.py --model='Baseline' --dataset='OfficeHome' --source='Clipart' --target='Real World' \
 --cuda='cuda:1' --num_workers=1 --iterations=10004  --test_interval=100 --batch_size=36


# DANN
python3.6 ../main.py --model='DANN' --dataset='OfficeHome' --source='Art' --target='Clipart' \
--cuda='cuda:1' --num_workers=0 --iterations=10004 --test_interval=100 --batch_size=36

python3.6 ../main.py --model='DANN' --dataset='OfficeHome' --source='Art' --target='Product'\
 --cuda='cuda:1' --num_workers=0 --iterations=10004  --test_interval=100 --batch_size=36

python3.6 ../main.py --model='DANN' --dataset='OfficeHome' --source='Art' --target='Real World' \
 --cuda='cuda:1' --num_workers=0 --iterations=10004  --test_interval=100 --batch_size=36

python3.6 ../main.py --model='DANN' --dataset='OfficeHome' --source='Clipart' --target='Art' \
--cuda='cuda:1' --num_workers=0 --iterations=10004 --test_interval=100 --batch_size=36

python3.6 ../main.py --model='DANN' --dataset='OfficeHome' --source='Clipart' --target='Product'\
 --cuda='cuda:1' --num_workers=0 --iterations=10004  --test_interval=100 --batch_size=36

python3.6 ../main.py --model='DANN' --dataset='OfficeHome' --source='Clipart' --target='Real World' \
 --cuda='cuda:1' --num_workers=0 --iterations=10004  --test_interval=100 --batch_size=36


# Baseline
python3.6 ../main.py --model='Baseline' --dataset='OfficeHome' --source='Product' --target='Art' \
--cuda='cuda:2' --num_workers=1  --iterations=4000 --test_interval=100 --batch_size=36

python3.6 ../main.py --model='Baseline' --dataset='OfficeHome' --source='Product' --target='Clipart'\
 --cuda='cuda:2' --num_workers=1  --iterations=4000  --test_interval=100 --batch_size=36

python3.6 ../main.py --model='Baseline' --dataset='OfficeHome' --source='Product' --target='Real World' \
 --cuda='cuda:2' --num_workers=1  --iterations=4000  --test_interval=100 --batch_size=36

python3.6 ../main.py --model='Baseline' --dataset='OfficeHome' --source='Real World' --target='Art' \
--cuda='cuda:2' --num_workers=1  --iterations=4000 --test_interval=100 --batch_size=36

python3.6 ../main.py --model='Baseline' --dataset='OfficeHome' --source='Real World' --target='Clipart'\
 --cuda='cuda:2' --num_workers=1  --iterations=4000  --test_interval=100 --batch_size=36

python3.6 ../main.py --model='Baseline' --dataset='OfficeHome' --source='Real World' --target='Product' \
 --cuda='cuda:2' --num_workers=1 --iterations=4000  --test_interval=100 --batch_size=36

# DANN

python3.6 ../main.py --model='DANN' --dataset='OfficeHome' --source='Product' --target='Art' \
--cuda='cuda:2' --num_workers=0  --iterations=4000 --test_interval=100 --batch_size=36

python3.6 ../main.py --model='DANN' --dataset='OfficeHome' --source='Product' --target='Clipart'\
 --cuda='cuda:2' --num_workers=0  --iterations=4000  --test_interval=100 --batch_size=36

python3.6 ../main.py --model='DANN' --dataset='OfficeHome' --source='Product' --target='Real World' \
 --cuda='cuda:2' --num_workers=0  --iterations=4000  --test_interval=100 --batch_size=36

python3.6 ../main.py --model='DANN' --dataset='OfficeHome' --source='Real World' --target='Art' \
--cuda='cuda:2' --num_workers=0  --iterations=4000 --test_interval=100 --batch_size=36

python3.6 ../main.py --model='DANN' --dataset='OfficeHome' --source='Real World' --target='Clipart'\
 --cuda='cuda:2' --num_workers=0  --iterations=4000  --test_interval=100 --batch_size=36

python3.6 ../main.py --model='DANN' --dataset='OfficeHome' --source='Real World' --target='Product' \
 --cuda='cuda:2' --num_workers=0  --iterations=4000  --test_interval=100 --batch_size=36









