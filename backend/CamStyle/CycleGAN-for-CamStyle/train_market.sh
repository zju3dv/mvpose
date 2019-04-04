# train all cameras for market

python train.py --dataroot ./datasets/market --name market-c1-c2 --camA 1 --camB 2
python train.py --dataroot ./datasets/market --name market-c1-c3 --camA 1 --camB 3
python train.py --dataroot ./datasets/market --name market-c1-c4 --camA 1 --camB 4
python train.py --dataroot ./datasets/market --name market-c1-c5 --camA 1 --camB 5
python train.py --dataroot ./datasets/market --name market-c1-c6 --camA 1 --camB 6

python train.py --dataroot ./datasets/market --name market-c2-c3 --camA 2 --camB 3
python train.py --dataroot ./datasets/market --name market-c2-c4 --camA 2 --camB 4
python train.py --dataroot ./datasets/market --name market-c2-c5 --camA 2 --camB 5
python train.py --dataroot ./datasets/market --name market-c2-c6 --camA 2 --camB 6

python train.py --dataroot ./datasets/market --name market-c3-c4 --camA 3 --camB 4
python train.py --dataroot ./datasets/market --name market-c3-c5 --camA 3 --camB 5
python train.py --dataroot ./datasets/market --name market-c3-c6 --camA 3 --camB 6

python train.py --dataroot ./datasets/market --name market-c4-c5 --camA 4 --camB 5
python train.py --dataroot ./datasets/market --name market-c4-c6 --camA 4 --camB 6

python train.py --dataroot ./datasets/market --name market-c5-c6 --camA 5 --camB 6
