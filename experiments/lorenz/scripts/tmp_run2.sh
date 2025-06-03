#! /bin/bash

cd ../

python train.py --config=train_win2 --epochs=3000
python train.py --config=train_win2_G --epochs=3000


python eval.py --config=eval_win2 --N_trajectory=64 --LT=15

python eval.py --config=eval_win2_G --N_trajectory=64 --LT=15

python eval.py --config=eval_win2_G_on_data --N_trajectory=64 --LT=15

python eval.py --config=eval_win2_on_Gdata --N_trajectory=64 --LT=15
