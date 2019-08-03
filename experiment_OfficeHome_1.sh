# Ar to Cl
python3.6 main.py --model='Baseline' --dataset='OfficeHome' --source='Art' --target='Clipart' \
--cuda='cuda:3' --num_workers=1 --iterations=10004 --test_interval=500 --batch_size=36

python3.6 main.py --model='MADA' --dataset='OfficeHome' --source='Art' --target='Clipart' \
--cuda='cuda:3' --num_workers=0 --iterations=10004 --test_interval=500 --batch_size=36

python3.6 main.py --model='DANN' --dataset='OfficeHome' --source='Art' --target='Clipart' \
--cuda='cuda:0' --num_workers=0 --iterations=10004 --test_interval=500 --batch_size=36

python3.6 main.py --model='MCD' --dataset='OfficeHome' --source='Art' --target='Clipart' \
--cuda='cuda:0' --num_workers=0 --iterations=10004 --test_interval=500 --batch_size=36



#python3.6 main.py --model='MT' --dataset='OfficeHome' --source='Art' --target='Clipart' \
#--cuda='cuda:1' --num_workers=0 --iterations=10004 --test_interval=500 --batch_size=36

# Cl to Pr
python3.6 main.py --model='Baseline' --dataset='OfficeHome' --source='Clipart' --target='Product'\
 --cuda='cuda:1' --num_workers=1 --iterations=10004  --test_interval=500 --batch_size=36

 python3.6 main.py --model='MADA' --dataset='OfficeHome' --source='Clipart' --target='Product'\
 --cuda='cuda:1' --num_workers=0 --iterations=10004  --test_interval=500 --batch_size=36

python3.6 main.py --model='DANN' --dataset='OfficeHome' --source='Clipart' --target='Product'\
 --cuda='cuda:1' --num_workers=0 --iterations=10004  --test_interval=500 --batch_size=36

python3.6 main.py --model='MCD' --dataset='OfficeHome' --source='Clipart' --target='Product'\
 --cuda='cuda:1' --num_workers=0 --iterations=10004  --test_interval=500 --batch_size=36



# python3.6 main.py --model='MT' --dataset='OfficeHome' --source='Clipart' --target='Product'\
# --cuda='cuda:1' --num_workers=0 --iterations=10004  --test_interval=100 --batch_size=36

 # Re to Pw
 python3.6 main.py --model='Baseline' --dataset='OfficeHome' --source='Real World' --target='Product' \
 --cuda='cuda:2' --num_workers=1  --iterations=10004  --test_interval=500 --batch_size=36

 python3.6 main.py --model='MADA' --dataset='OfficeHome' --source='Real World' --target='Product' \
 --cuda='cuda:2' --num_workers=0  --iterations=10004  --test_interval=500 --batch_size=36

 python3.6 main.py --model='DANN' --dataset='OfficeHome' --source='Real World' --target='Product' \
 --cuda='cuda:2' --num_workers=0  --iterations=10004  --test_interval=500 --batch_size=36


 python3.6 main.py --model='MCD' --dataset='OfficeHome' --source='Real World' --target='Product' \
 --cuda='cuda:2' --num_workers=0  --iterations=10004  --test_interval=500 --batch_size=36
 


# python3.6 main.py --model='MT' --dataset='OfficeHome' --source='Real World' --target='Product' \
# --cuda='cuda:2' --num_workers=0  --iterations=10004  --test_interval=100 --batch_size=36













python3.6 main.py --model='Baseline' --dataset='OfficeHome' --source='Art' --target='Product'\
 --cuda='cuda:1' --num_workers=1 --iterations=10004  --test_interval=100 --batch_size=36

python3.6 main.py --model='Baseline' --dataset='OfficeHome' --source='Art' --target='Real World' \
 --cuda='cuda:1' --num_workers=1 --iterations=10004  --test_interval=100 --batch_size=36

python3.6 main.py --model='Baseline' --dataset='OfficeHome' --source='Clipart' --target='Art' \
--cuda='cuda:1' --num_workers=1 --iterations=10004 --test_interval=100 --batch_size=36

python3.6 main.py --model='Baseline' --dataset='OfficeHome' --source='Clipart' --target='Product'\
 --cuda='cuda:1' --num_workers=1 --iterations=10004  --test_interval=100 --batch_size=36

python3.6 main.py --model='Baseline' --dataset='OfficeHome' --source='Clipart' --target='Real World' \
 --cuda='cuda:1' --num_workers=1 --iterations=10004  --test_interval=100 --batch_size=36


# DANN
python3.6 main.py --model='DANN' --dataset='OfficeHome' --source='Art' --target='Clipart' \
--cuda='cuda:1' --num_workers=0 --iterations=10004 --test_interval=100 --batch_size=36

python3.6 main.py --model='DANN' --dataset='OfficeHome' --source='Art' --target='Product'\
 --cuda='cuda:1' --num_workers=0 --iterations=10004  --test_interval=100 --batch_size=36

python3.6 main.py --model='DANN' --dataset='OfficeHome' --source='Art' --target='Real World' \
 --cuda='cuda:1' --num_workers=0 --iterations=10004  --test_interval=100 --batch_size=36

python3.6 main.py --model='DANN' --dataset='OfficeHome' --source='Clipart' --target='Art' \
--cuda='cuda:1' --num_workers=0 --iterations=10004 --test_interval=100 --batch_size=36

python3.6 main.py --model='DANN' --dataset='OfficeHome' --source='Clipart' --target='Product'\
 --cuda='cuda:1' --num_workers=0 --iterations=10004  --test_interval=100 --batch_size=36

python3.6 main.py --model='DANN' --dataset='OfficeHome' --source='Clipart' --target='Real World' \
 --cuda='cuda:1' --num_workers=0 --iterations=10004  --test_interval=100 --batch_size=36







