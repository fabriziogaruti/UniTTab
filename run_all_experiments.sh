#!/bin/bash

OUTPUT_DIR="./models_checkpoints/"
# export CUDA_VISIBLE_DEVICES=0



### POLLUTION DATASET ###

python main_pollution_lstm.py \
            --finetuning_output_dir=$OUTPUT_DIR \
            --finetuning_experiment_name="pollution_lstm_5" \
            --regression=true \
            --use_embeddings=true \
            --stride=5 \
            --batch_size=120
python main_pollution_finetuning.py \
            --finetuning_output_dir=$OUTPUT_DIR \
            --finetuning_experiment_name="pollution_finetuning_5" \
            --regression=true \
            --use_embeddings=true \
            --stride=5 \
            --batch_size=120

python main_pollution_lstm.py \
            --finetuning_output_dir=$OUTPUT_DIR \
            --finetuning_experiment_name="pollution_lstm_10" \
            --regression=true \
            --use_embeddings=true \
            --stride=10 \
            --batch_size=120
python main_pollution_finetuning.py \
            --finetuning_output_dir=$OUTPUT_DIR \
            --finetuning_experiment_name="pollution_finetuning_10" \
            --regression=true \
            --use_embeddings=true \
            --stride=10 \
            --batch_size=120

python main_pollution_lstm.py \
            --finetuning_output_dir=$OUTPUT_DIR \
            --finetuning_experiment_name="pollution_lstm_50" \
            --regression=true \
            --use_embeddings=true \
            --seq_len=50 \
            --stride=50 \
            --batch_size=120
python main_pollution_finetuning.py \
            --finetuning_output_dir=$OUTPUT_DIR \
            --finetuning_experiment_name="pollution_finetuning_50" \
            --regression=true \
            --use_embeddings=true \
            --seq_len=50 \
            --stride=50 \
            --batch_size=120



### FRAUD DATASET ###

python main_fraud_lstm.py \
            --data_fname="card_transaction.v1" \
            --finetuning_output_dir=$OUTPUT_DIR \
            --finetuning_experiment_name="fraud_lstm_10" \
            --regression=true \
            --use_embeddings=true

python main_fraud_lstm.py \
            --data_fname="card_transaction_50trans" \
            --finetuning_output_dir=$OUTPUT_DIR \
            --finetuning_experiment_name="fraud_lstm_50" \
            --regression=true \
            --use_embeddings=true \
            --batch_size=20



### AGE2 DATASET ###

python main_age2_lstm.py \
            --data_fname="transactions.parquet" \
            --finetuning_output_dir=$OUTPUT_DIR \
            --finetuning_experiment_name="age_lstm" \
            --regression=true \
            --use_embeddings=true \
            --batch_size=20
python main_age2_finetuning.py \
            --data_fname="transactions.parquet" \
            --finetuning_output_dir=$OUTPUT_DIR \
            --finetuning_experiment_name="age_finetuning" \
            --regression=true \
            --use_embeddings=true \
            --batch_size=20
