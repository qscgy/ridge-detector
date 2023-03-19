# CUDA_VISIBLE_DEVICES=0 python train_scribble.py \
#     --backbone mobilenet --lr 0.007 --workers 4 \
#     --base-dir /playpen/Datasets/scribble-samples/ \
#     --epochs 96 --batch-size 12  --out-stride 16 \
#     --checkname mobilenet4-96-all --eval-interval 1 \
#     --dataset pascal --save-interval 1 \
#     --densecrfloss 5e-09 --rloss-scale 0.5 \
#     --sigma-rgb-crf 15 --sigma-xy-crf 50 \
#     --ncloss 0 --sigma-rgb-nc 15 --sigma-xy-nc 40 \
#     --in-chan 4

# CUDA_VISIBLE_DEVICES=0 python train_scribble.py \
#     --backbone mobilenet --lr 0.007 --workers 4 \
#     --base-dir /playpen/Datasets/scribble-samples/ \
#     --epochs 96 --batch-size 12  --out-stride 16 \
#     --checkname mobilenet4-96-all --eval-interval 1 \
#     --dataset pascal --save-interval 1 \
#     --densecrfloss 5e-09 --rloss-scale 0.5 \
#     --sigma-rgb-crf 15 --sigma-xy-crf 50 \
#     --ncloss 0.5 --sigma-rgb-nc 15 --sigma-xy-nc 40 \
#     --in-chan 4

# CUDA_VISIBLE_DEVICES=0 python train_scribble.py \
#     --backbone mobilenet --lr 0.007 --workers 4 \
#     --base-dir /playpen/Datasets/scribble-samples/ \
#     --epochs 96 --batch-size 12  --out-stride 16 \
#     --checkname mobilenet4-96-all --eval-interval 1 \
#     --dataset pascal --save-interval 1 \
#     --densecrfloss 0 --rloss-scale 0.5 \
#     --sigma-rgb-crf 15 --sigma-xy-crf 50 \
#     --ncloss 0.5 --sigma-rgb-nc 15 --sigma-xy-nc 40 \
#     --in-chan 4

CUDA_VISIBLE_DEVICES=0 python train_scribble.py \
    --backbone mobilenet --lr 0.007 --workers 4 \
    --base-dir /playpen/Datasets/scribble-samples/ \
    --epochs 96 --batch-size 12  --out-stride 16 \
    --checkname mobilenet-normal --eval-interval 1 \
    --dataset pascal --save-interval 4 \
    --densecrfloss 0 --rloss-scale 0.5 \
    --sigma-rgb-crf 15 --sigma-xy-crf 50 --swirl 1.0 \
    --ncloss 0 --sigma-rgb-nc 15 --sigma-xy-nc 40 \
    --in-chan 6

# CUDA_VISIBLE_DEVICES=0 python train_scribble.py \
#     --backbone mobilenet --lr 0.007 --workers 4 \
#     --base-dir /playpen/Datasets/scribble-samples/ \
#     --epochs 96 --batch-size 12  --out-stride 16 \
#     --checkname mobilenet3-96-all --eval-interval 1 \
#     --dataset pascal --save-interval 1 \
#     --densecrfloss 5e-09 --rloss-scale 0.5 \
#     --sigma-rgb-crf 15 --sigma-xy-crf 50 \
#     --ncloss 0.5 --sigma-rgb-nc 15 --sigma-xy-nc 40 \
#     --in-chan 3

# CUDA_VISIBLE_DEVICES=0 python train_scribble.py \
#     --backbone mobilenet --lr 0.007 --workers 4 \
#     --base-dir /playpen/Datasets/scribble-samples/ \
#     --epochs 96 --batch-size 12  --out-stride 16 \
#     --checkname mobilenet3-96-all --eval-interval 1 \
#     --dataset pascal --save-interval 1 \
#     --densecrfloss 0 --rloss-scale 0.5 \
#     --sigma-rgb-crf 15 --sigma-xy-crf 50 \
#     --ncloss 0.5 --sigma-rgb-nc 15 --sigma-xy-nc 40 \
#     --in-chan 3

# CUDA_VISIBLE_DEVICES=0 python train_scribble.py \
#     --backbone mobilenet --lr 0.007 --workers 4 \
#     --base-dir /playpen/Datasets/scribble-samples/ \
#     --epochs 96 --batch-size 12  --out-stride 16 \
#     --checkname mobilenet3-96-all --eval-interval 1 \
#     --dataset pascal --save-interval 1 \
#     --densecrfloss 5e-09 --rloss-scale 0.5 \
#     --sigma-rgb-crf 15 --sigma-xy-crf 50 \
#     --ncloss 0 --sigma-rgb-nc 15 --sigma-xy-nc 40 \
#     --in-chan 3

# CUDA_VISIBLE_DEVICES=0 python train_scribble.py \
#     --backbone mobilenet --lr 0.007 --workers 4 \
#     --base-dir /playpen/Datasets/scribble-samples/ \
#     --epochs 96 --batch-size 12  --out-stride 16 \
#     --checkname mobilenet3-96-all --eval-interval 1 \
#     --dataset pascal --save-interval 1 \
#     --densecrfloss 0 --rloss-scale 0.5 \
#     --sigma-rgb-crf 15 --sigma-xy-crf 50 \
#     --ncloss 0 --sigma-rgb-nc 15 --sigma-xy-nc 40 \
#     --in-chan 3