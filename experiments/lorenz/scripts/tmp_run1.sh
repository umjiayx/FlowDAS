#! /bin/bash

cd ../

python train.py --config=train_win1 --epochs=3000
python train.py --config=train_win1_G --epochs=3000

python eval.py --config=eval_win1 --N_trajectory=64 --LT=15

python eval.py --config=eval_win1_G --N_trajectory=64 --LT=15

python eval.py --config=eval_win1_G_on_data --N_trajectory=64 --LT=15

python eval.py --config=eval_win1_on_Gdata --N_trajectory=64 --LT=15