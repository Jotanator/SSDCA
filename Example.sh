# export HUGGINGFACE_HUB_CACHE=./cluster/home/jtapias/custom_hf_cache

# Train Swin Small
# For 30 Epochs
# Using Fold 2
# Balanced sampling

python main.py --num_classes 2 \
--model_config Swin_S \
--num_epochs 30 \
--batch_size 8 \
--learning_rate 2e-5 \
--json_fold fold_2.json \
--sampler balanced \
--save_dir "finetunedmodel_allSameCrop_115" \
--model_naming Siam-Swin_S_Fold2_TemporalChange_115 \
--seed 115 \