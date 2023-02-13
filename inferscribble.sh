CUDA_VISIBLE_DEVICES=0 python infer_scribble.py \
    --backbone mobilenet --workers 4 --n_class 2 \
    --crop_size 216 --batch-size 1 \
    --checkpoint run/pascal/mobilenet-normal/ex_0_dorothy/checkpoint.pth.tar \
    --base-dir /playpen/Datasets/scribble-test/testA --in-chan 6 --figures
    # best are cyrus and lucille