#! /bin/bash

cd ../
python eval.py --config=eval_win1_G_on_data --N_trajectory=64 --LT=15
python eval.py --config=eval_win2_G_on_data --N_trajectory=64 --LT=15
python eval.py --config=eval_win3_G_on_data --N_trajectory=64 --LT=15