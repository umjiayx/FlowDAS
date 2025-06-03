#! /bin/bash

cd ../
python eval.py --config=eval_win1 --N_trajectory=64 --LT=15
python eval.py --config=eval_win2 --N_trajectory=64 --LT=15
python eval.py --config=eval_win3 --N_trajectory=64 --LT=15

