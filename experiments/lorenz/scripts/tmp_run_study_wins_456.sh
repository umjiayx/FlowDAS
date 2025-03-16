#! /bin/bash

cd ../


python train.py --config=train_win4_G
python train.py --config=train_win5_G
python train.py --config=train_win6_G

python eval.py --config=eval_win4_G
python eval.py --config=eval_win5_G
python eval.py --config=eval_win6_G

