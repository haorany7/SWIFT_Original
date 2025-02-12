#!/bin/bash

# Define variables for better readability and maintenance
SWIFT_OUTPUT="/scratch/bdes/haorany7/swift/SWIFT_original/SWIFT/outputs/cnndm/cnndm_100/model_answer/vicuna-7b-v1.5/vicuna-7b-v1.5-swift-float16-temp-0.0-top-p-0.85-seed-2024-max_new_tokens-512-opt_interval-1-bayes_interval-25-max_opt-1000-max_tolerance-300-max_score-0.93-context_window-50-skip_ratio-0.45.jsonl"
BASE_OUTPUT="/scratch/bdes/haorany7/swift/SWIFT_original/SWIFT/test/cnndm/cnndm_100/model_answer/vicuna-7b-v1.5/vicuna-7b-v1.5-vanilla-float16-temp-0.0-top-p-0.85-seed-2024-max_new_tokens-512.jsonl"
TOKENIZER="lmsys/vicuna-7b-v1.5"
# Run the evaluation
python evaluation_llama/speed.py \
    --file-path "$SWIFT_OUTPUT" \
    --base-path "$BASE_OUTPUT" \
    --tokenizer-path "$TOKENIZER"