for dataset in anli wnli rte cb sick ethos-binary ethos-religion ethos-gender ethos-race ethos-national_origin ethos-violence ethos-disability tweet-hate tweet-offensive tweet-irony tweet-feminist tweet-atheism sbic hate-speech18 trec subj agnews dbpedia financial-phrasebank poem-sentiment mr cr sst2
do
    CUDA_VISIBLE_DEVICES=0,1
    python run_classification.py \
    --model "EleutherAI/gpt-j-6b" \
    --dataset $dataset \
    --num_seeds 1 \
    --all_shots 2 \
    --subsample_test_set 10 \
    --approx \
    --lambda1 0.5 \
done