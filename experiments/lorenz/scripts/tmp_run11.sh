#! /bin/bash

cd ../


python train.py --config=train_win10

python eval.py --config=eval_win10

python eval.py --config=eval_win10_on_Gdata