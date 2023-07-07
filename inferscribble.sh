num="035"

CUDA_VISIBLE_DEVICES=0 python infer_scribble.py \
    --backbone mobilenet --workers 4 --n_class 2 \
    --crop_size 216 --extension png --batch-size 1 \
    --checkpoint run/pascal/mobilenet3-96/ex_0_cyrus/checkpoint.pth.tar \
    --base-dir /bigpen/simulator_data/LEC_fast \
    --in-chan 3 --outdir /bigpen/simulator_data/LEC_fast \
    # --use-examples \
    # --foldit-path True \
    #  --gt pickles/annotations_test_2.pkl
    # best are cyrus and lucille
