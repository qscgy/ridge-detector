CUDA_VISIBLE_DEVICES=0 python train_scribble.py \
    --backbone resnet --lr 0.007 --workers 4 \
    --base-dir /playpen/Datasets/scribble-samples/ \
    --epochs 60 --batch-size 12  \
    --checkname deeplab-resnet --eval-interval 2 \
    --dataset pascal --save-interval 2 \
    --densecrfloss 2e-9 --rloss-scale 0.5 \
    --sigma-rgb 15 --sigma-xy 100
