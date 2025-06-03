#! /bin/bash

cd ../
python eval.py --config=eval_win1_G --N_trajectory=64 --LT=15
python eval.py --config=eval_win2_G --N_trajectory=64 --LT=15
python eval.py --config=eval_win3_G --N_trajectory=64 --LT=15
