#! /bin/bash

cd ../

python train.py --config=train_win3_G --epochs=3000

python eval.py --config=eval_win3_G --N_trajectory=64 --LT=15

python eval.py --config=eval_win3_G_on_data --N_trajectory=64 --LT=15