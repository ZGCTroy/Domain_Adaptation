python3 main.py --source='Amazon' --target='Webcam' --cuda='cpu' --epochs=2 --if_test --batch_size=10 --num_workers=0

python3 main.py --source='Amazon' --target='Dslr' --cuda='cuda:0' --epochs=2 --if_test --batch_size=36 --num_workers=0

python3 main.py --source='Dslr' --target='Webcam' --cuda='cuda:0' --epochs=2 --if_test --batch_size=36 --num_workers=0

python3 main.py --source='Dslr' --target='Amazon' --cuda='cuda:0' --epochs=2 --if_test --batch_size=36 --num_workers=0 

python3 main.py --source='Webcam' --target='Dslr' --cuda='cuda:0' --epochs=2 --if_test --batch_size=36 --num_workers=0 

python3 main.py --source='Webcam' --target='Amazon' --cuda='cuda:0' --epochs=2 --if_test --batch_size=36 --num_workers=0 
 
