# for dataset in anli wnli rte cb sick ethos-binary ethos-religion ethos-gender ethos-race ethos-national_origin ethos-violence ethos-disability tweet-hate tweet-offensive tweet-irony tweet-feminist tweet-atheism civil-comments sbic hate-speech18 trec subj agnews dbpedia financial-phrasebank poem-sentiment mr cr sst2 gutenberg-time 
# do
#     CUDA_VISIBLE_DEVICES=0,1
#     python run_classification.py \
#     --model "EleutherAI/gpt-neo-2.7B" \
#     --dataset $dataset \
#     --num_seeds 5 \
#     --all_shots 8 \
#     --subsample_test_set 500 \
#     --approx \
#     --replace_ratio '0.5' 
# done

# for dataset in anli wnli rte cb sick ethos-binary ethos-religion ethos-gender ethos-race ethos-national_origin ethos-violence ethos-disability tweet-hate tweet-offensive tweet-irony tweet-feminist tweet-atheism civil-comments sbic hate-speech18 trec subj agnews dbpedia financial-phrasebank poem-sentiment mr cr sst2 gutenberg-time 
# do
#     CUDA_VISIBLE_DEVICES=0,1
#     python run_classification.py \
#     --model "EleutherAI/gpt-neo-2.7B" \
#     --dataset $dataset \
#     --num_seeds 5 \
#     --all_shots 8 \
#     --subsample_test_set 500 \
#     --approx \
#     --replace_ratio '0.5' \
#     --do_task_learning
# done

for dataset in agnews trec financial-phrasebank dbpedia cb sick anli tweet-stance_atheism tweet-stance_feminist
do
    CUDA_VISIBLE_DEVICES=0,1
    python run_classification.py \
    --model "EleutherAI/gpt-j-6b" \
    --dataset $dataset \
    --num_seeds 10 \
    --all_shots 8 \
    --subsample_test_set 20 \
    --approx \
    --replace_ratio '0.5' \
    --perform_chaining_effect
done