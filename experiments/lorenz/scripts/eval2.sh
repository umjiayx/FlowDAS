#! /bin/bash

cd ../
python eval.py --config=eval_win1_on_Gdata --N_trajectory=64 --LT=15
python eval.py --config=eval_win2_on_Gdata --N_trajectory=64 --LT=15
python eval.py --config=eval_win3_on_Gdata --N_trajectory=64 --LT=15