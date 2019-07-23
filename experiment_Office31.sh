
# Baseline

python3.6 main.py --model='Baseline' --dataset='Office31' --source='Amazon' --target='Webcam' \
--cuda='cuda:1' --num_workers=1  --iterations=10004 --test_interval=100 --batch_size=36

python3.6 main.py --model='Baseline' --dataset='Office31' --source='Amazon' --target='Dslr'\
 --cuda='cuda:1' --num_workers=1  --iterations=10004  --test_interval=100 --batch_size=36

python3.6 main.py --model='Baseline' --dataset='Office31' --source='Dslr' --target='Webcam' \
 --cuda='cuda:1' --num_workers=1  --iterations=10004  --test_interval=100 --batch_size=36

python3.6 main.py --model='Baseline' --dataset='Office31' --source='Dslr' --target='Amazon' \
--cuda='cuda:1' --num_workers=1  --iterations=10004 --test_interval=100 --batch_size=36

python3.6 main.py --model='Baseline' --dataset='Office31' --source='Webcam' --target='Dslr' \
--cuda='cuda:1' --num_workers=1  --iterations=10004  --test_interval=100 --batch_size=36

python3.6 main.py --model='Baseline' --dataset='Office31' --source='Webcam' --target='Amazon' \
--cuda='cuda:1' --num_workers=1  --iterations=10004  --test_interval=100 --batch_size=36



# DANN

python3.6 main.py --model='DANN' --dataset='Office31' --source='Amazon' --target='Webcam' \
--cuda='cuda:1' --num_workers=0  --iterations=10004 --test_interval=100 --batch_size=36

python3.6 main.py --model='DANN' --dataset='Office31' --source='Amazon' --target='Dslr'\
 --cuda='cuda:1' --num_workers=0  --iterations=10004  --test_interval=100 --batch_size=36

python3.6 main.py --model='DANN' --dataset='Office31' --source='Dslr' --target='Webcam' \
 --cuda='cuda:1' --num_workers=0  --iterations=10004  --test_interval=100 --batch_size=36

python3.6 main.py --model='DANN' --dataset='Office31' --source='Dslr' --target='Amazon' \
--cuda='cuda:1' --num_workers=0  --iterations=10004 --test_interval=100 --batch_size=36

python3.6 main.py --model='DANN' --dataset='Office31' --source='Webcam' --target='Dslr' \
--cuda='cuda:1' --num_workers=0  --iterations=10004  --test_interval=100 --batch_size=36

python3.6 main.py --model='DANN' --dataset='Office31' --source='Webcam' --target='Amazon' \
--cuda='cuda:1' --num_workers=0  --iterations=10004  --test_interval=100 --batch_size=36



