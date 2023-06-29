CUDA_VISIBLE_DEVICES=0,1
python run_classification.py \
--model "EleutherAI/gpt-j-6b" \
--dataset "ethos-binary" \
--num_seeds 5 \
--all_shots 8 \
--subsample_test_set 500 \
--approx 