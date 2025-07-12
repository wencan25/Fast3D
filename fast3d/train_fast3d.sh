# nohup bash train_fast3d.sh > outputs/training_fast3d_run1.log 2>&1 &
accelerate launch training_fast3d.py --config config/train.yaml