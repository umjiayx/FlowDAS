#! /bin/bash

cd ../


python train.py --config=train_win10_G

python eval.py --config=eval_win10_G

python eval.py --config=eval_win10_G_on_data