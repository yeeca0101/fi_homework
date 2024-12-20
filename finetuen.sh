CUDA_VISIBLE_DEVICES=0,1 python3 train.py --lr 5e-6 --batch_size 32  \
    --gpus 2 \
    --optim adam \
    --log_dir logs \
    --epoch 300 \
    --loss ssim \
    --arch attnv4 \
    --monitor_metric ssim \
    --train_mode last \
    --alpha 0.4 \
    --all_mask True \
    --post_fix all_mask_ramdom_pre \
    --weight_decay 1e-3 \
    --finetune True \
    --save_folder checkpoint/attnv4/ssim/all_mask_ramdom_pre