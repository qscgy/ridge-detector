CUDA_VISIBLE_DEVICES=0 python train_scribble.py \
    --backbone mobilenet --lr 0.007 --workers 4 \
    --base-dir /playpen/Datasets/scribble-samples/ \
    --epochs 40 --batch-size 12  --out-stride 16 \
    --checkname mobilenet3-v2 --eval-interval 1 \
    --dataset pascal --save-interval 1 \
    --densecrfloss 0 --rloss-scale 0.5 \
    --sigma-rgb-crf 15 --sigma-xy-crf 50 \
    --ncloss 0.5 --sigma-rgb-nc 15 --sigma-xy-nc 40 \
    --in-chan 3
