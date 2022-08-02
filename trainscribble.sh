CUDA_VISIBLE_DEVICES=0 python train_scribble.py \
    --backbone mobilenet --lr 0.007 --workers 4 \
    --base-dir /playpen/Datasets/scribble-samples/ \
    --epochs 40 --batch-size 12  \
    --checkname deeplab-mobilenet --eval-interval 2 \
    --dataset pascal --save-interval 4 \
    --densecrfloss 0 --rloss-scale 0.5 \
    --sigma-rgb-crf 15 --sigma-xy-crf 20 \
