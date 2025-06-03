#! /bin/bash

cd ../

python train.py --config=train_win3 --epochs=3000

python eval.py --config=eval_win3 --N_trajectory=64 --LT=15

python eval.py --config=eval_win3_on_Gdata --N_trajectory=64 --LT=15