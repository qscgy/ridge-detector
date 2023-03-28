CUDA_VISIBLE_DEVICES=0 python infer_scribble.py \
    --backbone mobilenet --workers 4 --n_class 2 \
    --crop_size 216 --batch-size 1 \
    --checkpoint run/pascal/mobilenet3-96/ex_0_cyrus/checkpoint.pth.tar \
    --base-dir /bigpen/Datasets/jhu-older/cecum1a-2/image --in-chan 3
    # best are cyrus and lucille