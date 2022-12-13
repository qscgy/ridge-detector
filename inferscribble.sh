CUDA_VISIBLE_DEVICES=0 python infer_scribble.py \
    --backbone mobilenet --workers 4 --n_class 2 \
    --crop_size 216 --batch-size 1 \
    --checkpoint run/pascal/deeplab-mobilenet4-v2/ex_1_richard/checkpoint.pth.tar \
    --base-dir /playpen/Datasets/scribble-test/testA --in-chan 4 \
    --use-examples