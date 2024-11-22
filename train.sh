
# random pre train : 5e-4, 32, ssim, 0.8, 1e-3 800
# fine yet
CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 1e-3 --batch_size 32  \
    --gpus 1 \
    --optim adam \
    --log_dir logs \
    --epoch 300 \
    --loss ssim \
    --arch attnv4 \
    --monitor_metric ssim \
    --train_mode last \
    --alpha 0.1 \
    --all_mask True \
    --post_fix test \
    --weight_decay 1e-3 \
    --save_folder test
    # --finetune True \
