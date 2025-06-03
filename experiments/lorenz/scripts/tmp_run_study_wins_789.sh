#! /bin/bash

cd ../


python train.py --config=train_win7_G
python train.py --config=train_win8_G
python train.py --config=train_win9_G


python eval.py --config=eval_win7_G
python eval.py --config=eval_win8_G
python eval.py --config=eval_win9_G