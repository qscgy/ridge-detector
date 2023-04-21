num="019"

CUDA_VISIBLE_DEVICES=0 python infer_scribble.py \
    --backbone mobilenet --workers 4 --n_class 2 \
    --crop_size 216 --batch-size 1 \
    --checkpoint run/pascal/mobilenet3-96/ex_0_cyrus/checkpoint.pth.tar \
    --base-dir /playpen/Datasets/geodepth2/$num/image \
    --in-chan 3 --outdir /playpen/Datasets/geodepth2/$num
    # best are cyrus and lucille
