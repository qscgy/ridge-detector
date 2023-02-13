CUDA_VISIBLE_DEVICES=0 python infer_scribble.py \
    --backbone mobilenet --workers 4 --n_class 2 \
    --crop_size 216 --batch-size 1 \
    --checkpoint run/pascal/mobilenet4-96/ex_4_lucille/checkpoint.pth.tar \
    --base-dir /playpen/Datasets/scribble-samples/annotations --in-chan 4 --figures \
    --sequence-dir /playpen/Datasets/geodepth2/ \
    --foldit-dir /playpen/CEP/results/foldit_public/test_latest/images \

    # best are cyrus and lucille