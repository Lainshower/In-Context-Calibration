CUDA_VISIBLE_DEVICES=0,1
python run_classification.py \
--model "EleutherAI/gpt-neo-125m" \
--dataset "cb" \
--num_seeds 5 \
--all_shots 8 \
--subsample_test_set 5 \
--approx \
--num_contexts 2 \
--num_of_calibration 5 10 20 \ 